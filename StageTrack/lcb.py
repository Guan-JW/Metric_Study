from multiprocessing import Value
import os
import argparse
import numpy as np
import pandas as pd

import sys
sys.path.append("../")
from LCBench.api import Benchmark
from utils import *
import ConfigSpace as CS


parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--config_num', type=int, default=1000, help='Rounds of running')
parser.add_argument('--store_dir', type=str, default='./rst/', help='The directory to put results')
parser.add_argument('--dataset', type=str, default='adult', help='Dataset name. Candidates are: adult, higgs, jasmine, Fashion-MNIST, vehicle, volkert]')
args = parser.parse_args()

dataset = args.dataset
config_num = args.config_num
dir_path = args.store_dir
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
# Add dataset to the directory name
dir_path = os.path.join(dir_path, dataset)
if not os.path.exists(dir_path):
	os.makedirs(dir_path)

bench_dir = "../LCBench/data/six_datasets_lw.json"
bench = Benchmark(bench_dir, cache=False)

total_config_num = bench.get_number_of_configs(dataset)
string = CS.CategoricalHyperparameter('str', [str(i) for i in range(total_config_num)])
seed = np.random.randint(1, 100000)
config_space = CS.ConfigurationSpace(seed=seed)
config_space.add_hyperparameters([string])

data_seed = 777
max_iter = 50

# files that record losses of each configuration at each epoch
val_loss_file = os.path.join(dir_path, "val_loss.csv")
train_loss_file = os.path.join(dir_path, "train_loss.csv")
# files that record the derivatives of losses for each configuration at each epoch
val_loss_deriv_file = os.path.join(dir_path, "val_loss_deriv.csv")
train_loss_deriv_file = os.path.join(dir_path, "train_loss_deriv.csv")

val_loss = {}
train_loss = {}
val_loss_deriv = {}
train_loss_deriv = {}
for c in range(config_num):
	# Randomly sample one configuration
	config = bench.sample_config(dataset)
    # Initialize
	val_loss[config] = np.zeros(max_iter)
	train_loss[config] = np.zeros(max_iter)
	val_loss_deriv[config] = np.zeros(max_iter - 1)
	train_loss_deriv[config] = np.zeros(max_iter - 1)
	# Loop over 
	for epoch in range(max_iter):
		fidelity = epoch
		vloss = bench.query(dataset, tag='Train/val_cross_entropy', config_id=config)[fidelity]
		tloss = bench.query(dataset, tag='Train/train_cross_entropy', config_id=config)[fidelity]
		val_loss[config][epoch - 1] = vloss
		train_loss[config][epoch - 1] = tloss
		
		if epoch == 1:
			continue
		val_loss_deriv[config][epoch - 2] = (vloss - val_loss[config][epoch - 2])
		train_loss_deriv[config][epoch - 2] = (tloss - train_loss[config][epoch - 2])

files = [val_loss_file, train_loss_file, val_loss_deriv_file, train_loss_deriv_file]
for i, data in enumerate([val_loss, train_loss, val_loss_deriv, train_loss_deriv]):
	file = files[i]
	# Convert the dictionary into a list of tuples
	data_tuples = [(key, *values) for key, values in data.items()]
	df = pd.DataFrame(data_tuples)
	df.to_csv(file, index=False, header=False)
	print(f"Data is written to {file} successfully.")
	