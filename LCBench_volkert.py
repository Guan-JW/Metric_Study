import json
import warnings
from Schemes.sh import SuccessiveHalving
from Schemes.hyperband import Hyperband
from LCBench.api import Benchmark
import numpy as np
import os
import re
import argparse
import numpy as np
import pandas as pd
from utils import *
import ConfigSpace as CS

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--task', type=str, default='acc_loss', help='Description of task')
parser.add_argument('--rounds', type=int, default=1000, help='Rounds of running')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy', 'test_losses'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[9, 15, 30, 45], help='List of integers (default: [10, 15, 30, 50])')
parser.add_argument('--eta', type=float, default=3.0, help='Fraction of saving in hyperband')
parser.add_argument('--scheme', type=str, default='Hyperband', help='HPO scheme: Hyperband or BOHB')
args = parser.parse_args()

max_iters = args.max_iters
ult_objs = args.ult_objs
task = args.task
rounds = args.rounds
eta = args.eta
if eta.is_integer():
	eta = int(eta)
dataset_name = 'volkert'
dataset = dataset_name
scheme = args.scheme


bench_dir = "LCBench/data/six_datasets_lw.json"
bench = Benchmark(bench_dir, cache=False)

total_config_num = bench.get_number_of_configs(dataset_name)
string = CS.CategoricalHyperparameter('str', [str(i) for i in range(total_config_num)])
seed = np.random.randint(1, 100000)
config_space = CS.ConfigurationSpace(seed=seed)
config_space.add_hyperparameters([string])


criterias, direction = get_cta_dir(task)

def get_params():
	return bench.sample_config(dataset_name)


def try_params(n_iteration, config_id, criteria='valid_accuracy'):
	print("n_iteration: ", n_iteration)
	print("criteria: ", criteria)
	print(f"{config_id=}")
	fidelity = round(n_iteration)

	rtn_dic = dict()
	rtn_dic['time'] = bench.query(dataset_name=dataset_name, tag="time", config_id=config_id)[fidelity]
	rtn_dic['test_accuracy'] = bench.query(dataset_name=dataset_name, tag="final_test_accuracy", config_id=config_id)
	rtn_dic['test_losses'] = bench.query(dataset_name, tag='final_test_cross_entropy', config_id=config_id)

	# get final validation or training precision & losses
	if 'valid' in criteria:
		rtn_dic['final_valid_accuracy'] = bench.query(dataset_name, tag='final_test_accuracy', config_id=config_id)
		rtn_dic['final_valid_loss'] = bench.query(dataset_name, tag='final_test_cross_entropy', config_id=config_id)
	elif 'train' in criteria:
		rtn_dic['final_train_accuracy'] = bench.query(dataset_name, tag='final_val_accuracy', config_id=config_id)
		rtn_dic['final_train_loss'] = bench.query(dataset_name, tag='final_val_cross_entropy', config_id=config_id)
	
	if 'stage' in criteria:
		match = re.search(r'\d+', criteria)
		if match:
			stage = int(match.group())
		else:
			print("No number found in the string.")
			exit(0)
		if check_stage(dataset, fidelity, stage): # should use validation loss
			print(f"Using validation loss at epoch {fidelity}")
			if 'win' in criteria:
				val_arr = []
				window_size = get_window_size(dataset)
				if round(n_iteration) < window_size:
					for i in range(1, round(n_iteration) + 2):
						hist_fidelity = i - 1
						if 'train_loss' in criteria:
							val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
						else:
							val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
				else:
					for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
						hist_fidelity = i - 1
						if 'train_loss' in criteria:
							val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
						else:
							val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
						
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
		else:
			print(f"Using training loss at epoch {fidelity}")
			if 'win' in criteria:
				val_arr = []
				window_size = get_window_size(dataset)
				if round(n_iteration) < window_size:
					for i in range(1, round(n_iteration) + 2):
						hist_fidelity = i - 1
						val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
				else:
					for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
						hist_fidelity = i - 1
						val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
						
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[fidelity]
	
	elif 'win_mu' in criteria:
		window_size = get_window_size(dataset)
		val_arr = []
		if round(n_iteration) < window_size:
			for i in range(1, round(n_iteration) + 2):
				hist_fidelity = i - 1
				if 'valid_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[hist_fidelity])
				elif 'valid_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
				elif 'train_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[hist_fidelity])
				elif 'train_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
		else:
			for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
				hist_fidelity = i - 1
				if 'valid_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[hist_fidelity])
				elif 'valid_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
				elif 'train_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[hist_fidelity])
				elif 'train_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		rtn_dic[criteria] = mu
	elif 'loss_change_rate' in criteria:
		last_fidelity = fidelity - 1
		if 'valid' in criteria:
			cnt_val_loss = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
			last_val_loss = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[last_fidelity]
			rcr = 0 - (cnt_val_loss - last_val_loss)   # minimize
			rtn_dic['valid_losses'] = cnt_val_loss
		elif 'train' in criteria:
			cnt_train_loss = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[fidelity]
			last_train_loss = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[last_fidelity]
			rcr = 0 - (cnt_train_loss - last_train_loss)
			rtn_dic['train_losses'] = cnt_train_loss
		rtn_dic[criteria] = rcr
	
	elif 'seeds' in criteria:
		if 'valid_accuracy' in criteria:
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[fidelity]
		elif 'valid_losses' in criteria:
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
	
	elif criteria.startswith('win_head'):
		val_arr = []
		for i in range(1, round(n_iteration) + 2):
			hist_fidelity = i - 1
			if 'valid_accuracy' in criteria:
				val_arr.append(bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[hist_fidelity])
			elif 'valid_loss' in criteria:
				val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
			elif 'train_accuracy' in criteria:
				val_arr.append(bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[hist_fidelity])
			elif 'train_loss' in criteria:
			   val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		sig = np.std(val_arr)
		
		if 'mu+sig' in criteria:
			rtn_dic[criteria] = mu + sig
		elif 'mu-sig' in criteria:
			rtn_dic[criteria] = mu - sig
		elif 'mu' in criteria:
			rtn_dic[criteria] = mu
		else:
			print("Unknown criteria")
			exit(0)
	
	elif 'win' in criteria:
		# uncertainty-related values
		match = re.search(r'\d+', criteria)
		if match:
			window_size = int(match.group())
		else:
			print("No number found in the string.")
			exit(0)
		val_arr = []
		if round(n_iteration) < window_size:
			for i in range(1, round(n_iteration) + 2):
				hist_fidelity = i - 1
				if 'valid_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[hist_fidelity])
				elif 'valid_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
				elif 'train_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[hist_fidelity])
				elif 'train_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
		else:
			for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
				hist_fidelity = i - 1
				if 'valid_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[hist_fidelity])
				elif 'valid_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[hist_fidelity])
				elif 'train_accuracy' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[hist_fidelity])
				elif 'train_loss' in criteria:
					val_arr.append(bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[hist_fidelity])
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		sig = np.std(val_arr)

		if criteria.startswith('dyn_wgh'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_criteria(mu, sig, n_iteration, 49)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_criteria(mu, -sig, n_iteration, 49)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_sig'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_sigma_criteria(mu, sig, n_iteration, 49)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_sigma_criteria(mu, -sig, n_iteration, 49)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_log_sig'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_log_sigma_criteria(mu, sig, n_iteration, 49)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_log_sigma_criteria(mu, -sig, n_iteration, 49)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_exp'):
			pattern = r'dyn_exp_(\d+)'
			match = re.search(pattern, criteria)
			if match:
				factor = int(match.group(1))
			else:
				raise ValueError(f"No exp number given in criteria {criteria}.")
			
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_exp_sigma_criteria(mu, sig, n_iteration, 1/factor)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_exp_sigma_criteria(mu, -sig, n_iteration, 1/factor)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_sqrt'):
			pattern = r'dyn_sqrt_(\d+)'
			match = re.search(pattern, criteria)
			if match:
				factor = int(match.group(1))
			else:
				raise ValueError(f"No exp number given in criteria {criteria}.")
			
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_sqrt_sigma_criteria(mu, sig, n_iteration, factor)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_sqrt_sigma_criteria(mu, -sig, n_iteration, factor)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif 'dyn' in criteria:
			rtn_dic[criteria] = (mu, sig)
			return rtn_dic

		elif criteria.startswith('fix'):
			pattern = r'fix_(\d+)_'
			match = re.search(pattern, criteria)
			if match:
				fix_epoch = int(match.group(1))
			else:
				raise ValueError(f"No epoch number given in criteria {criteria}.")
			
			if n_iteration < fix_epoch:
				if 'accuracy' in criteria:
					rtn_dic[criteria] = mu + sig
				elif 'loss' in criteria:
					rtn_dic[criteria] = mu - sig
				else:
					raise ValueError(f"Neither loss or accuracy in criteria {criteria}.")
			else:
				rtn_dic[criteria] = mu
			return rtn_dic

		if 'mu+sig' in criteria:
			rtn_dic[criteria] = mu + sig
		elif 'mu-sig' in criteria:
			rtn_dic[criteria] = mu - sig
		elif 'mu' in criteria:
			rtn_dic[criteria] = mu
		else:
			print("Unknown criteria")
			exit(0)

	elif criteria.startswith('wgh'):
		''' !!!! Direction is minimize by default !!!! '''

		# 1. Calculate loss change rate
		last_fidelity = fidelity - 1
		if 'valid_lcr' in criteria:
			cnt_val_loss = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
			last_val_loss = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[last_fidelity]
			first_val = 0 - (cnt_val_loss - last_val_loss)
			if 'valid_lcr_min' in criteria:
				first_val = 0 - first_val
			rtn_dic['valid_losses'] = cnt_val_loss
		elif 'train_lcr' in criteria:
			cnt_train_loss = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[fidelity]
			last_train_loss = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[last_fidelity]
			first_val = 0 - (cnt_train_loss - last_train_loss)
			if 'train_lcr_min' in criteria:
				first_val = 0 - first_val
			rtn_dic['train_losses'] = cnt_train_loss
		else:
			raise ValueError("Not defined criteria.")

		# 2. Get the second value
		if 'valid_loss' in criteria:
			second_val = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
		elif 'valid_accuracy' in criteria:
			# Minimize, so directly pass precision !!
			second_val = 100 - bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[fidelity]
		elif 'train_loss' in criteria:
			second_val = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[fidelity]
		elif 'train_accuracy' in criteria:
			# Minimize, so directly pass precision !!
			second_val = 100 - bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[fidelity]

		rtn_dic[criteria] = (first_val, second_val)
		# print(first_val, second_val)
		# exit()

	else:
		if criteria == 'train_accuracy':
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/val_accuracy', config_id=config_id)[fidelity]
		elif criteria == 'train_losses':
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/val_cross_entropy', config_id=config_id)[fidelity]
		elif criteria == 'valid_accuracy':
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/test_accuracy', config_id=config_id)[fidelity]
		elif criteria == 'valid_losses':
			rtn_dic[criteria] = bench.query(dataset_name, tag='Train/test_cross_entropy', config_id=config_id)[fidelity]
		else:
			print(f"Unknown criteria {criteria}")
			exit(0)
		
	return rtn_dic


# Initialize criteria dictionary
Ult_obj_dict = dict()
for obj in ult_objs:
	Ult_obj_dict[obj] = dict()

for max_iter in max_iters:
	print(f"################## max_iter = {max_iter} ######################")
	# Initialize criteria dictionary
	for obj in ult_objs:
		if 'accuracy' in obj:
			Ult_obj_dict[obj]['Max_test_accuracy'] = []
		if 'loss' in obj:
			Ult_obj_dict[obj]['Min_test_loss'] = []

		for criteria in criterias:
			Ult_obj_dict[obj][criteria] = []
	
	for r in range(rounds):
		print(f"########### round = {r} ############")

		# Run HPO scheme, the configurations are the same for testing each criteria
		if scheme == 'Hyperband':
			hb = Hyperband(get_params, try_params, max_iter=max_iter, eta=eta)
		elif scheme == 'SH':
			hb = SuccessiveHalving(get_params, try_params, max_iter=max_iter, eta=eta)
		else:
			raise ValueError(f"Unkonwn HPO scheme: {scheme}")

		# Load or generate fixed configurations
		config_dir = os.path.join(f"Records/{scheme}/", dataset, str(hb))
		if not os.path.exists(config_dir):
			os.makedirs(config_dir)
		config_dir = os.path.join(config_dir, "fixed_configs")
		
		config_file = os.path.join(config_dir, "config" + str(r) + ".json")
		config_file_exist = False
		if os.path.exists(config_file):
			hb.load_fixed_config_dict_lcbench(config_file)
			config_file_exist = True
		else:
			warnings.warn(f"File {config_file} doesn't exist, generate fixed configurations.")

		for i, criteria in enumerate(criterias):
			print(f"########### criteria = {criteria} ############")
			 # hb.run()
			rst = hb.run_fixed_configs(criteria=criteria, direction=direction[i])

			# Record results
			dir = os.path.join(f"Records/{scheme}/", dataset, str(hb), "config_rsts")
			if not os.path.exists(dir):
				os.makedirs(dir)
			dir = os.path.join(dir, criteria)
			if not os.path.exists(dir):
				os.makedirs(dir)
			file = os.path.join(dir, "record" + str(r) + ".csv")
			hb.record_to_csv(rst, record_file=file)

			# Get the best configuration selected by hyperband algorithm
			rst = pd.DataFrame(rst)
			best_rst = rst.iloc[-1]
			for obj in ult_objs:
				if 'accuracy' in obj:
					Ult_obj_dict[obj][criteria].append(best_rst['test_accuracy'])
				if 'loss' in obj:
					Ult_obj_dict[obj][criteria].append(best_rst['test_losses'])
			if i == 0:
				# Must get fixed configuration after one round of hyperband
				if config_file_exist == False:
					fixed_config_dic = hb.get_fixed_config_dict_lcbench()
					if not os.path.exists(config_dir):
						os.makedirs(config_dir)
					# Write configurations to file
					with open(config_file, "w") as json_file:
						json.dump(fixed_config_dic, json_file)

				# Get overall statistics of current group of configuration
				for obj in ult_objs:
					if 'accuracy' in obj:
						Ult_obj_dict[obj]['Max_test_accuracy'].append(rst['test_accuracy'].max())
					if 'loss' in obj:
						Ult_obj_dict[obj]['Min_test_loss'].append(rst['test_losses'].min())
		
		for obj in ult_objs:
			warnings.warn(f'Ult_obj_dict[{obj}] = {Ult_obj_dict[obj]}')

	# Store obtained test values from different criterias into file 
	for obj in ult_objs:
		dir = os.path.join(f"Records/{scheme}/", dataset, str(hb), "cta")
		if not os.path.exists(dir):
			os.makedirs(dir)
		dir = os.path.join(dir, f"obj_{obj}")
		if not os.path.exists(dir):
			os.makedirs(dir)
		file = os.path.join(dir, f"{task}.csv")
		warnings.warn(f"file = {file}")
		df = pd.DataFrame(Ult_obj_dict[obj])
		df.to_csv(file, index=False)