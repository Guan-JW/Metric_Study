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
from utils import *
import random

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--data', type=str, default="7592", help='Dataset number')
parser.add_argument('--task', type=str, default='loss', help='Description of task')
parser.add_argument('--rounds', type=int, default=1000, help='Rounds of running')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy', 'test_losses'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[25, 50, 75, 100], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=float, default=3, help='Fraction of saving for earch stopping')
parser.add_argument('--scheme', type=str, default='Hyperband', help='HPO scheme: Hyperband or BOHB')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

max_iters = args.max_iters
ult_objs = args.ult_objs
task = args.task
rounds = args.rounds
dataset = f'NN_{args.data}'
scheme = args.scheme
eta = args.eta
if isinstance(eta, float) and eta.is_integer():
	eta = int(eta)
seed = args.seed

dir_path = os.path.join("/home/SSD/HPO/Empirical_study/StageTrack/rst", dataset, str(seed))
val_loss_file = os.path.join(dir_path, "val_loss.csv")
train_loss_file = os.path.join(dir_path, "train_loss.csv")
test_acc_file = os.path.join(dir_path, "test_acc.csv")
test_loss_file = os.path.join(dir_path, "test_loss.csv")
df_val_loss = pd.read_csv(val_loss_file, header=None).iloc[:, 1:]
df_train_loss = pd.read_csv(train_loss_file, header=None).iloc[:, 1:]
df_test_acc = pd.read_csv(test_acc_file, header=None).iloc[:, 1:]
assert df_train_loss.shape[0] == df_val_loss.shape[0]
assert df_train_loss.shape[0] == df_test_acc.shape[0]
total_sample_num = df_train_loss.shape[0]

if 'seed' in args.task:
	data_seed = get_seeds(dataset)
	for s in data_seed:
		dir_path = os.path.join("/home/SSD/HPO/Empirical_study/StageTrack/rst", dataset, f"{s}")
		val_loss_file = os.path.join(dir_path, "val_loss.csv")
		train_loss_file = os.path.join(dir_path, "train_loss.csv")
		test_acc_file = os.path.join(dir_path, "test_acc.csv")
		df_val_loss += pd.read_csv(val_loss_file, header=None).iloc[:, 1:]
		df_train_loss += pd.read_csv(train_loss_file, header=None).iloc[:, 1:]
		df_test_acc += pd.read_csv(test_acc_file, header=None).iloc[:, 1:]
	
	df_val_loss /= len(data_seed)
	df_train_loss /= len(data_seed)
	df_test_acc /= len(data_seed)

criterias, direction = get_cta_dir(task)


def get_params():
	return random.randint(0, total_sample_num - 1)

def try_params(n_iteration, config, criteria='valid_accuracy', last_epoch=0):
	fidelity = {'epoch': round(n_iteration) + 1 + 25}	# NN fidelity range [1, 200]
	final_fidelity = {'epoch': 243}	# NN fidelity range [1, 243]

	# Update learning curves
	last_epoch = 0
	new_config_curve_dict = dict()
	new_config_curve_dict["valid_loss"] = df_val_loss.iloc[config, last_epoch : fidelity['epoch'] + 1].values.tolist()
	new_config_curve_dict["train_loss"] = df_train_loss.iloc[config, last_epoch: fidelity['epoch'] + 1].values.tolist()
	new_config_curve_dict["cnt_epoch"] = [fidelity['epoch']]
	
	result_dic = {
		'info': {
			'train_losses': df_train_loss.iloc[config, fidelity['epoch'] - 1],
			'valid_losses': df_val_loss.iloc[config, fidelity['epoch'] - 1]
		}
	}

	rtn_dic = {'time': 0,
			   'test_accuracy': df_test_acc.iloc[config, 0]
			   }
	
	if 'stage_adp_loss' in criteria:
		# Update std & derivatives
		# Get stage info
		stage = get_derivative_std_stage(new_config_curve_dict, get_window_size(dataset))		
		if stage == 1 or stage in ['1', '2.1']:
			rtn_dic[criteria] = new_config_curve_dict["train_loss"][-1]
		elif stage in ['2.2', '3']:
			rtn_dic[criteria] = new_config_curve_dict["valid_loss"][-1]
		else:
			if check_stage(dataset, fidelity['epoch'], 3):
				rtn_dic[criteria] = new_config_curve_dict["valid_loss"][-1]
			else:
				rtn_dic[criteria] = new_config_curve_dict["train_loss"][-1]

	elif 'stage' in criteria:
		match = re.search(r'\d+', criteria)
		if match:
			stage = int(match.group())
		else:
			print("No number found in the string.")
			exit(0)
		if check_stage(dataset, fidelity['epoch'], stage): # should use validation loss
			if 'win' in criteria:
				window_size = get_window_size(dataset)
				val_arr = []
				if round(n_iteration) < window_size:
					val_arr += df_val_loss.iloc[config, 1 : round(n_iteration) + 2].values.tolist()
				else:
					val_arr += df_val_loss.iloc[config, round(n_iteration) - window_size + 2 : round(n_iteration) + 2].values.tolist()
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = result_dic['info']['valid_losses']
			# already averaged value when using seed!
		else:
			if 'win' in criteria:
				window_size = get_window_size(dataset)
				val_arr = []
				if round(n_iteration) < window_size:
					val_arr += df_train_loss.iloc[config, 1 : round(n_iteration) + 2].values.tolist()
				else:
					val_arr += df_train_loss.iloc[config, round(n_iteration) - window_size + 2 : round(n_iteration) + 2].values.tolist()
				val_arr = np.array(val_arr)
				mu = np.mean(val_arr)
				rtn_dic[criteria] = mu
			else:
				rtn_dic[criteria] = result_dic['info']['train_losses']
			
	elif 'win_mu' in criteria:
		window_size = get_window_size(dataset)
		val_arr = []
		if round(n_iteration) < window_size:
			if 'valid_loss' in criteria:
				val_arr += df_val_loss.iloc[config, 1 : round(n_iteration) + 2].values.tolist()
			elif 'train_loss' in criteria:
				val_arr += df_train_loss.iloc[config, 1 : round(n_iteration) + 2].values.tolist()
		else:
			if 'valid_loss' in criteria:
				val_arr += df_val_loss.iloc[config, round(n_iteration) - window_size + 2 : round(n_iteration) + 2].values.tolist()
			elif 'train_loss' in criteria:
				val_arr += df_train_loss.iloc[config, round(n_iteration) - window_size + 2 : round(n_iteration) + 2].values.tolist()
		val_arr = np.array(val_arr)
		mu = np.mean(val_arr)
		rtn_dic[criteria] = mu
			
	elif 'seed' in criteria:
		if 'accuracy' in criteria:
			raise ValueError("Do not support accuracy currently.")
		elif 'valid_losses' in criteria:
			rtn_dic[criteria] = result_dic['info']['valid_losses']
		elif 'train_losses' in criteria:
			rtn_dic[criteria] = result_dic['info']['train_losses']
   
	elif 'accuracy' in criteria:
		raise ValueError("No training/validation accuracy available.")
	elif 'loss' in criteria and 'losses' not in criteria:
		bench_cta = criteria.replace("loss", "losses")
		rtn_dic[criteria] = result_dic['info'][bench_cta]
	else:
		rtn_dic[criteria] = result_dic['info'][criteria]

	return rtn_dic, new_config_curve_dict


# Initialize criteria dictionary
Ult_obj_dict = dict()
for obj in ult_objs:
	Ult_obj_dict[obj] = dict()

for max_iter in max_iters:
	# Initialize criteria dictionary
	for obj in ult_objs:
		if 'accuracy' in obj:
			Ult_obj_dict[obj]['Max_test_accuracy'] = []
		if 'loss' in obj:
			Ult_obj_dict[obj]['Min_test_loss'] = []

		for criteria in criterias:
			Ult_obj_dict[obj][criteria] = []

	for r in range(rounds):
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
			hb.load_fixed_int_config_dict(config_file)
			config_file_exist = True
		else:
			warnings.warn(f"File {config_file} doesn't exist, generate fixed configurations.")

		for i, criteria in enumerate(criterias):
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
					fixed_config_dic = hb.get_fixed_int_config_dict()
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