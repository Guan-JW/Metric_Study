from multiprocessing import Value
import os
import re
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import sys
from scipy.misc import derivative
sys.path.append("../")

from DataLoader.test_pybnn import BNNOnToyFunction_HPO, BNNOnBostonHousing_HPO, BNNOnProteinStructure_HPO, BNNOnYearPrediction_HPO
from utils import *

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--data', type=str, default="Boston", help='Dataset name')
parser.add_argument('--config_num', type=int, default=1000, help='Rounds of running')
parser.add_argument('--seed', type=int, default=1, help='Rounds of running')
parser.add_argument('--store_dir', type=str, default='./rst/', help='The directory to put results')
args = parser.parse_args()

dataset = f"Pybnn_{args.data}"
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

if "Toy" in dataset:
	bnn = BNNOnToyFunction_HPO(rng=seed)
elif "Boston" in dataset:
	bnn = BNNOnBostonHousing_HPO(rng=seed)
elif "Protein" in dataset:
	bnn = BNNOnProteinStructure_HPO(rng=seed)
elif "Year" in dataset:
	bnn = BNNOnYearPrediction_HPO(rng=seed)

config_space = bnn.get_configuration_space(seed=seed)
burn_in_steps = 1
max_iter = 10000 - burn_in_steps

# files that record losses of each configuration at each epoch
val_loss_file = os.path.join(dir_path, "val_loss.csv")
train_loss_file = os.path.join(dir_path, "train_loss.csv")
test_loss_file = os.path.join(dir_path, "test_loss.csv")
# files that record the derivatives of losses for each configuration at each epoch
val_loss_deriv_file = os.path.join(dir_path, "val_loss_deriv.csv")
train_loss_deriv_file = os.path.join(dir_path, "train_loss_deriv.csv")

val_loss = {}
train_loss = {}
test_loss = {}
val_loss_deriv = {}
train_loss_deriv = {}
for c in range(config_num):
	# Randomly sample one configuration
	config = config_space.sample_configuration()
	# repeat till no repetition
	while config in val_loss.keys():
		config = config_space.sample_configuration()
    # Initialize
	val_loss_deriv[config] = np.zeros(max_iter - 1)
	train_loss_deriv[config] = np.zeros(max_iter - 1)

	fidelity = {'budget': max_iter}
	result_dic = bnn.objective_function_test(configuration=config, fidelity=fidelity, rng=seed)
	val_loss[config] = np.asarray(result_dic['valid_losses'])
	train_loss[config] = np.asarray(result_dic['train_losses'])
	test_loss[config] = np.asarray(result_dic['test_losses'])

	for epoch in range(max_iter):
		if epoch == 0:
			continue
		val_loss_deriv[config][epoch - 1] = (val_loss[config][epoch] - val_loss[config][epoch - 1])
		train_loss_deriv[config][epoch - 1] = (train_loss[config][epoch] - train_loss[config][epoch - 1])

	files = [val_loss_file, train_loss_file, test_loss_file, val_loss_deriv_file, train_loss_deriv_file]
	for i, data in enumerate([val_loss, train_loss, test_loss, val_loss_deriv, train_loss_deriv]):
		file = files[i]
		data_tuples = [(str(key), *values) for key, values in data.items()]
		df = pd.DataFrame(data_tuples)
		df.to_csv(file, mode='a', index=False, header=False)
	
	val_loss.clear()
	train_loss.clear()
	test_loss.clear()
	val_loss_deriv.clear()
	train_loss_deriv.clear()
	