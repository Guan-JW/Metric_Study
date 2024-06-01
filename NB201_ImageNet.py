from multiprocessing import Value
import os
import re
import json
from unittest import result
import warnings
import argparse
import numpy as np
import pandas as pd
from Schemes.sh import SuccessiveHalving
from Schemes.hyperband import Hyperband
from DataLoader.test_nasbench_201 import *
from utils import *


parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--task', type=str, default='acc_loss', help='Description of task')
parser.add_argument('--rounds', type=int, default=1000, help='Rounds of running')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy', 'test_losses'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[50, 81, 160, 180], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=float, default=3.0, help='Fraction of saving for earch stopping')
parser.add_argument('--scheme', type=str, default='Hyperband', help='HPO scheme: Hyperband or BOHB')
args = parser.parse_args()

max_iters = args.max_iters
ult_objs = args.ult_objs
task = args.task
rounds = args.rounds
dataset = 'ImageNet'
scheme = args.scheme
eta = args.eta
if eta.is_integer():
	eta = int(eta)

NB201_ImageNet = ImageNetNasBench201Benchmark(rng=1)
config_space = NB201_ImageNet.get_configuration_space(seed=1)

criterias, direction = get_cta_dir(task)

def get_params():
	return config_space.sample_configuration()


def try_params(n_iteration, config, criteria='valid_accuracy'):
	print("n_iteration: ", n_iteration)
	print("criteria: ", criteria)
	fidelity = {'epoch': round(n_iteration) + 1}	# Nasbench_201 fidelity range [1, 200]
	final_fidelity = {'epoch': 200}	# Nasbench_201 fidelity range [1, 200]

	# If more than one seed is given, the results are averaged 
			# across the seeds but then the training time is the sum of the costs per seed.
	data_seed=777
	if 'seed' in criteria:
		data_seed = (777, 888, 999)

	result_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=fidelity, data_seed=data_seed)
	final_rst_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=final_fidelity, data_seed=data_seed)
	rtn_dic = {'time': result_dic['cost'],
			   'test_accuracy': 100 - result_dic['info']['test_precision'], 
			   'test_losses': result_dic['info']['test_losses']}
	
	# get final validation or training accuracy & losses
	if 'valid' in criteria:
		rtn_dic['final_valid_accuracy'] = 100 - final_rst_dic['info']['valid_precision']
		rtn_dic['final_valid_loss'] = final_rst_dic['info']['valid_losses']
	elif 'train' in criteria:
		rtn_dic['final_train_accuracy'] = 100 - final_rst_dic['info']['train_precision']
		rtn_dic['final_train_loss'] = final_rst_dic['info']['train_losses']
	
	if 'stage' in criteria:
		match = re.search(r'\d+', criteria)
		if match:
			stage = int(match.group())
		else:
			print("No number found in the string.")
			exit(0)
		if check_stage(dataset, fidelity['epoch'], stage): # should use validation loss
			print(f"Using validation loss at epoch {fidelity['epoch']}")
			if 'win' in criteria:
				window_size = get_window_size(dataset)
				val_arr = []
				if round(n_iteration) < window_size:
					for i in range(1, round(n_iteration) + 2):
						hist_fidelity = {'epoch': i}
						hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
						val_arr.append(hist_dic['info']['valid_losses']) # validation!!!
				else:
					for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
						hist_fidelity = {'epoch': i}
						hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
						val_arr.append(hist_dic['info']['valid_losses']) # validation!!!
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = result_dic['info']['valid_losses']
			# already averaged value when using seed!
		else:
			print(f"Using training loss at epoch {fidelity['epoch']}")
			if 'win' in criteria:
				window_size = get_window_size(dataset)
				val_arr = []
				if round(n_iteration) < window_size:
					for i in range(1, round(n_iteration) + 2):
						hist_fidelity = {'epoch': i}
						hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
						val_arr.append(hist_dic['info']['train_losses'])	# Train!!!
				else:
					for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
						hist_fidelity = {'epoch': i}
						hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
						val_arr.append(hist_dic['info']['train_losses'])	# Train !!!
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = result_dic['info']['train_losses']
			
	elif 'win_mu' in criteria:
		window_size = get_window_size(dataset)
		val_arr = []
		if round(n_iteration) < window_size:
			for i in range(1, round(n_iteration) + 2):
				hist_fidelity = {'epoch': i}
				hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
				if 'valid_loss' in criteria:
					val_arr.append(hist_dic['info']['valid_losses'])
				elif 'train_loss' in criteria:
					val_arr.append(hist_dic['info']['train_losses'])
		else:
			for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
				hist_fidelity = {'epoch': i}
				hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
				if 'valid_loss' in criteria:
					val_arr.append(hist_dic['info']['valid_losses'])
				elif 'train_loss' in criteria:
					val_arr.append(hist_dic['info']['train_losses'])
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		rtn_dic[criteria] = mu
			
	elif 'loss_change_rate' in criteria:
		'''Maximize lcr by default'''
		last_fidelity = {'epoch': round(n_iteration)}
		last_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=last_fidelity, data_seed=data_seed)
		if 'valid' in criteria:
			rcr = result_dic['info']['valid_losses'] - last_dic['info']['valid_losses']
			rtn_dic['valid_losses'] = result_dic['info']['valid_losses']
		elif 'train' in criteria:
			rcr = result_dic['info']['train_losses'] - last_dic['info']['train_losses']
			rtn_dic['train_losses'] = result_dic['info']['train_losses']
		rtn_dic[criteria] = rcr
   
	elif 'seed' in criteria:
		if 'valid_accuracy' in criteria:
			rtn_dic[criteria] = 100 - result_dic['info']['valid_precision']
		elif 'valid_losses' in criteria:
			rtn_dic[criteria] = result_dic['info']['valid_losses']
		elif 'train_accuracy' in criteria:
			rtn_dic[criteria] = 100 - result_dic['info']['train_precision']
		elif 'train_losses' in criteria:
			rtn_dic[criteria] = result_dic['info']['train_losses']
   
	elif criteria.startswith('win_head'):
		val_arr = []
		for i in range(1, round(n_iteration) + 2):
			hist_fidelity = {'epoch': i}
			hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
			if 'valid_accuracy' in criteria:
				val_arr.append(100 - hist_dic['info']['valid_precision'])
			elif 'valid_loss' in criteria:
				val_arr.append(hist_dic['info']['valid_losses'])
			elif 'train_accuracy' in criteria:
				val_arr.append(100 - hist_dic['info']['train_precision'])
			elif 'train_loss' in criteria:
				val_arr.append(hist_dic['info']['train_losses'])
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
				hist_fidelity = {'epoch': i}
				hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
				if 'valid_accuracy' in criteria:
					val_arr.append(100 - hist_dic['info']['valid_precision'])
				elif 'valid_loss' in criteria:
					val_arr.append(hist_dic['info']['valid_losses'])
				elif 'train_accuracy' in criteria:
					val_arr.append(100 - hist_dic['info']['train_precision'])
				elif 'train_loss' in criteria:
					val_arr.append(hist_dic['info']['train_losses'])
		else:
			for i in range(round(n_iteration) - window_size + 2, round(n_iteration) + 2):
				hist_fidelity = {'epoch': i}
				hist_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=hist_fidelity, data_seed=data_seed)
				if 'valid_accuracy' in criteria:
					val_arr.append(100 - hist_dic['info']['valid_precision'])
				elif 'valid_loss' in criteria:
					val_arr.append(hist_dic['info']['valid_losses'])
				elif 'train_accuracy' in criteria:
					val_arr.append(100 - hist_dic['info']['train_precision'])
				elif 'train_loss' in criteria:
					val_arr.append(hist_dic['info']['train_losses'])
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		sig = np.std(val_arr)
			
		if criteria.startswith('dyn_wgh'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_criteria(mu, sig, n_iteration, 199)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_criteria(mu, -sig, n_iteration, 199)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_sig'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_sigma_criteria(mu, sig, n_iteration, 199)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_sigma_criteria(mu, -sig, n_iteration, 199)
			else:
				raise ValueError(f"Neither accuracy nor loss in criteria {criteria}.")
			return rtn_dic
		
		elif criteria.startswith('dyn_log_sig'):
			if 'accuracy' in criteria:
				rtn_dic[criteria] = dynamic_log_sigma_criteria(mu, sig, n_iteration, 199)
			elif 'loss' in criteria:
				rtn_dic[criteria] = dynamic_log_sigma_criteria(mu, -sig, n_iteration, 199)
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
		
		elif criteria.startswith('dyn'):
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
		last_fidelity = {'epoch': round(n_iteration)}
		last_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=last_fidelity, data_seed=data_seed)
		if 'valid_lcr' in criteria:
			first_val = 0 - (result_dic['info']['valid_losses'] - last_dic['info']['valid_losses'])
			if 'valid_lcr_min' in criteria:
				first_val = 0 - first_val
			rtn_dic['valid_losses'] = result_dic['info']['valid_losses']
		elif 'train_lcr' in criteria:
			first_val = 0 - (result_dic['info']['train_losses'] - last_dic['info']['train_losses'])
			if 'train_lcr_min' in criteria:
				first_val = 0 - first_val
			rtn_dic['train_losses'] = result_dic['info']['train_losses']
		else:
			raise ValueError("Not defined criteria.")

		# 2. Get the second value
		if 'valid_loss' in criteria:
			second_val = result_dic['info']['valid_losses']
		elif 'valid_accuracy' in criteria:
			# Minimize, so directly pass precision !!
			second_val = result_dic['info']['valid_precision']
		elif 'train_loss' in criteria:
			second_val = result_dic['info']['train_losses']
		elif 'train_accuracy' in criteria:
			# Minimize, so directly pass precision !!
			second_val = result_dic['info']['train_precision']

		rtn_dic[criteria] = (first_val, second_val)
	
	elif criteria.startswith('dyn_lin_train_val'):
		if 'accuracy' in criteria:
			train_cta = 100 - result_dic['info']['train_precision']
			val_cta = 100 - result_dic['info']['valid_precision']
		elif 'loss' in criteria:
			train_cta = result_dic['info']['train_losses']
			val_cta = result_dic['info']['valid_losses']
		rtn_dic[criteria] = dynamic_train_val(train_cta, val_cta, n_iteration, 199)
	
	elif criteria.startswith('dyn_log_train_val'):
		if 'accuracy' in criteria:
			train_cta = 100 - result_dic['info']['train_precision']
			val_cta = 100 - result_dic['info']['valid_precision']
		elif 'loss' in criteria:
			train_cta = result_dic['info']['train_losses']
			val_cta = result_dic['info']['valid_losses']
		rtn_dic[criteria] = dynamic_log_train_val(train_cta, val_cta, n_iteration, 199)
		
	elif criteria.startswith('dyn_exp_train_val'):
		if 'accuracy' in criteria:
			train_cta = 100 - result_dic['info']['train_precision']
			val_cta = 100 - result_dic['info']['valid_precision']
		elif 'loss' in criteria:
			train_cta = result_dic['info']['train_losses']
			val_cta = result_dic['info']['valid_losses']
		rtn_dic[criteria] = dynamic_exp_train_val(train_cta, val_cta, n_iteration)
	
	elif 'accuracy' in criteria:
		bench_cta = criteria.replace("accuracy", "precision")
		rtn_dic[criteria] = 100 - result_dic['info'][bench_cta]
	elif 'loss' in criteria and 'losses' not in criteria:
		bench_cta = criteria.replace("loss", "losses")
		rtn_dic[criteria] = result_dic['info'][bench_cta]
	else:
		rtn_dic[criteria] = result_dic['info'][criteria]

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
		if not os.path.exists(os.path.join(f"/home/SSD/HPO/Empirical_study/Records/{scheme}/", dataset)):
			os.makedirs(os.path.join(f"/home/SSD/HPO/Empirical_study/Records/{scheme}/", dataset))
		config_dir = os.path.join(f"/home/SSD/HPO/Empirical_study/Records/{scheme}/", dataset, str(hb))
		if not os.path.exists(config_dir):
			os.makedirs(config_dir)
		config_dir = os.path.join(config_dir, "fixed_configs")
		
		config_file = os.path.join(config_dir, "config" + str(r) + ".json")
		config_file_exist = False
		if os.path.exists(config_file):
			hb.load_fixed_config_dict(config_file, config_space)
			config_file_exist = True
		else:
			warnings.warn(f"File {config_file} doesn't exist, generate fixed configurations.")

		for i, criteria in enumerate(criterias):
			print(f"########### criteria = {criteria} ############")
			rst = hb.run_fixed_configs(criteria=criteria, direction=direction[i])

			# Record results
			dir = os.path.join(f"/home/SSD/HPO/Empirical_study/Records/{scheme}/", dataset, str(hb), "config_rsts")
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
					fixed_config_dic = hb.get_fixed_config_dict(config_space)
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
		dir = os.path.join(f"/home/SSD/HPO/Empirical_study/Records/{scheme}/", dataset, str(hb), "cta")
		if not os.path.exists(dir):
			os.makedirs(dir)
		dir = os.path.join(dir, f"obj_{obj}")
		if not os.path.exists(dir):
			os.makedirs(dir)
		file = os.path.join(dir, f"{task}.csv")
		warnings.warn(f"file = {file}")
		df = pd.DataFrame(Ult_obj_dict[obj])
		df.to_csv(file, index=False)
