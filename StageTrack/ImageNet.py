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
from DataLoader.test_nasbench_201 import *
from utils import *


parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--config_num', type=int, default=1000, help='Rounds of running')
parser.add_argument('--store_dir', type=str, default='./rst/', help='The directory to put results')
args = parser.parse_args()

dataset = 'ImageNet'
config_num = args.config_num
dir_path = args.store_dir
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
# Add dataset to the directory name
dir_path = os.path.join(dir_path, dataset)
if not os.path.exists(dir_path):
	os.makedirs(dir_path)

NB201_ImageNet = ImageNetNasBench201Benchmark(rng=1)
config_space = NB201_ImageNet.get_configuration_space(seed=1)
data_seed = 777
max_iter = 200

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
	config = config_space.sample_configuration()
    # Initialize
	val_loss[config] = np.zeros(max_iter)
	train_loss[config] = np.zeros(max_iter)
	val_loss_deriv[config] = np.zeros(max_iter - 1)
	train_loss_deriv[config] = np.zeros(max_iter - 1)
	# Loop over 
	for epoch in range(1, max_iter + 1):
		fidelity = {'epoch': round(epoch)}	# Nasbench_201 fidelity range [1, 200]
		result_dic = NB201_ImageNet.objective_function(configuration=config, fidelity=fidelity, data_seed=data_seed)
		vloss = result_dic['info']['valid_losses']
		tloss = result_dic['info']['train_losses']
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
	