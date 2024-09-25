from multiprocessing import Value
import os
import re
import json
from unittest import result
import warnings
import argparse
import numpy as np
import pandas as pd
import sys
from scipy.misc import derivative
sys.path.append("../")

from DataLoader.test_nn import NNBenchmark_HPO
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
from utils import *

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--config_num', type=int, default=1000, help='Rounds of running')
parser.add_argument('--store_dir', type=str, default='./rst/', help='The directory to put results')
parser.add_argument('--task_id', type=int, default=258, help='Task id')
parser.add_argument('--seed', type=int, default=1, help='Rounds of running')
args = parser.parse_args()

dataset = f'NN_{args.task_id}'
config_num = args.config_num
dir_path = args.store_dir
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
# Add dataset to the directory name
dir_path = os.path.join(dir_path, dataset)
if not os.path.exists(dir_path):
	os.makedirs(dir_path)

seed = args.seed
store_dir = os.path.join(dir_path, dataset, str(seed))
if not os.path.exists(store_dir):
	os.makedirs(store_dir)

# task id
nn = NNBenchmark_HPO(args.task_id)
config_space = nn.get_configuration_space(seed=seed)
max_iter = 243

# files that record losses of each configuration at each epoch
val_loss_file = os.path.join(dir_path, "val_loss.csv")
train_loss_file = os.path.join(dir_path, "train_loss.csv")
test_loss_file = os.path.join(dir_path, "test_loss.csv")
test_acc_file = os.path.join(dir_path, "test_acc.csv")
# files that record the derivatives of losses for each configuration at each epoch
val_loss_deriv_file = os.path.join(dir_path, "val_loss_deriv.csv")
test_loss_deriv_file = os.path.join(dir_path, "test_loss_deriv.csv")
train_loss_deriv_file = os.path.join(dir_path, "train_loss_deriv.csv")

val_loss = {}
train_loss = {}
test_loss = {}
test_acc = {}
val_loss_deriv = {}
test_loss_deriv = {}
train_loss_deriv = {}
for c in range(config_num):
	# Randomly sample one configuration
	config = config_space.sample_configuration()
	# repeat till no repetition
	while config in val_loss.keys():
		config = config_space.sample_configuration()
    # Initialize
	val_loss_deriv[config] = np.zeros(max_iter - 1)
	test_loss_deriv[config] = np.zeros(max_iter - 1)
	train_loss_deriv[config] = np.zeros(max_iter - 1)

	fidelity = {'iter': max_iter}
	result_dic = nn.objective_function(configuration=config, fidelity=fidelity, rng=seed)
	val_loss[config] = np.asarray(result_dic['valid_losses'])
	test_loss[config] = np.asarray(result_dic['test_losses'])
	train_loss[config] = np.asarray(result_dic['train_losses'])
	test_acc[config] = [result_dic['test_accuracy']]

	for epoch in range(max_iter):
		if epoch == 0:
			continue
		val_loss_deriv[config][epoch - 1] = (val_loss[config][epoch] - val_loss[config][epoch - 1])
		test_loss_deriv[config][epoch - 1] = (test_loss[config][epoch] - test_loss[config][epoch - 1])
		train_loss_deriv[config][epoch - 1] = (train_loss[config][epoch] - train_loss[config][epoch - 1])

	files = [val_loss_file, train_loss_file, test_loss_file, test_acc_file, val_loss_deriv_file, test_loss_deriv_file, train_loss_deriv_file]
	for i, data in enumerate([val_loss, train_loss, test_loss, test_acc, val_loss_deriv, test_loss_deriv, train_loss_deriv]):
		file = files[i]
		data_tuples = [(str(key), *values) for key, values in data.items()]
		df = pd.DataFrame(data_tuples)
		df.to_csv(file, mode='a', index=False, header=False)
	
	val_loss.clear()
	train_loss.clear()
	test_loss.clear()
	test_acc.clear()
	val_loss_deriv.clear()
	test_loss_deriv.clear()
	train_loss_deriv.clear()