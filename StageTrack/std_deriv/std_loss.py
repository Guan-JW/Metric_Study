import csv
from math import nan
import sys
sys.path.append("../")
sys.path.append("../../")
import json
# from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the color map module
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

import pandas as pd
import os
import argparse
import numpy as np

import ruptures as rpt

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--store_dir_lcb', type=str, default='../../StageTrack/rst_w_test/', help='The directory to put results')
parser.add_argument('--store_dir_nb201', type=str, default='../../StageTrack/rst/', help='The directory to put results')
parser.add_argument('--detect_point', type=bool, default=False, help='If we should find change points or not')
args = parser.parse_args()

datasets = [
	"Cifar10", "Cifar100", "ImageNet", 
	"Fashion-MNIST", "volkert", "NN_146606",
	"NN_7592", "Pybnn_Boston", "Pybnn_Protein"
	]

label_dic = {"Cifar10": "CIFAR-10",
			 "Cifar100": "CIFAR-100",
			 "ImageNet": "ImageNet-16-120",
			 "volkert": "Volkert",
			 "NN_146606": "Higgs",
			 "NN_7592": "Adult",
			 "Pybnn_Boston": "Boston",
			 "Pybnn_Protein": "Protein"}

fig = plt.figure(figsize=(6.5, 6))
outer = GridSpec(3, 3, wspace=0.25, hspace=0.5)

n_configs = 6
start = 672
starts = [398, 479, 230]
fontsize=11
for i, dataset in enumerate(datasets):
	max_iter = 50
	window_size = 3 # 
	dir_path = os.path.join(args.store_dir_lcb, dataset)
	n_bkps_val = 2
	n_bkps_train = 2
	if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
		max_iter = 200
		window_size = 5
		dir_path = os.path.join(args.store_dir_nb201, dataset)
		n_bkps_val = 3
	elif "NN" in dataset:
		max_iter = 243
		window_size = 5
		dir_path = os.path.join(args.store_dir_nb201, dataset, '10')
		n_bkps_train = 3
		n_bkps_val = 3
	elif "Pybnn" in dataset:
		max_iter = 10000
		window_size = 5
		dir_path = os.path.join(args.store_dir_nb201, dataset)
		n_bkps_train = 10
		n_bkps_val = 10
	elif "OE" in dataset:
		max_iter = 101
		window_size = 5
		dir_path = os.path.join(args.store_dir_nb201, dataset)
		n_bkps_train = 3

	# files that record the derivatives of losses for each configuration at each epoch
	val_loss_deriv_file = os.path.join(dir_path, "val_loss.csv")
	train_loss_deriv_file = os.path.join(dir_path, "train_loss.csv")
	df_val_loss_deriv = pd.read_csv(val_loss_deriv_file, header=None)
	df_train_loss_deriv = pd.read_csv(train_loss_deriv_file, header=None)
	config_num_val = df_val_loss_deriv.shape[0]
	config_num_train = df_train_loss_deriv.shape[0]
	# check iteration number 
	max_iter = df_val_loss_deriv.shape[1] - 1
	assert max_iter == df_train_loss_deriv.shape[1] - 1
	max_iter = df_train_loss_deriv.shape[1] - 1
	assert config_num_val == config_num_train
	
	config_num = config_num_val
	window_cnt = round((max_iter - 1) / window_size)
	# remove the first column of the dfs, because they are configuration names / ids
	df_val_loss_deriv = df_val_loss_deriv.iloc[:, 1:]
	df_train_loss_deriv = df_train_loss_deriv.iloc[:, 1:]

	val_deriv_stds = np.zeros((config_num, max_iter - window_size * 2 + 1))
	train_deriv_stds = np.zeros((config_num, max_iter - window_size * 2 + 1))
	df_val_loss_deriv_array = df_val_loss_deriv.to_numpy()
	df_train_loss_deriv_array = df_train_loss_deriv.to_numpy()
	# Loop over to calculate stds.
	for c in range(config_num):
		val_deriv_stds[c, :] = pd.Series(df_val_loss_deriv_array[c]).rolling(window=window_size).std().fillna(0).to_numpy()[window_size:(max_iter - window_size + 1)]
		train_deriv_stds[c, :] = pd.Series(df_train_loss_deriv_array[c]).rolling(window=window_size).std().fillna(0).to_numpy()[window_size:(max_iter - window_size + 1)]
	
	def detect_change_points(signal, penalty=3, model="linear", n_bkps=2):
		if model == "linear":
			algo = rpt.Dynp(model="clinear").fit(signal)
		elif model == "rbf":
			algo = rpt.Dynp(model="rbf").fit(signal)
		result = algo.predict(n_bkps=n_bkps)
		return result
	
	df_val_stds = pd.DataFrame(val_deriv_stds)
	df_train_stds = pd.DataFrame(train_deriv_stds)
	mean_val_stds = df_val_stds.mean().T
	mean_train_stds = df_train_stds.mean().T	
	smoothed_mean_val_stds = mean_val_stds.rolling(window=window_size, min_periods=1).mean()
	smoothed_mean_train_stds = mean_train_stds.rolling(window=window_size, min_periods=1).mean()
	# stds
	if args.detect_point:
		change_points_val_stds = detect_change_points(mean_val_stds.values, n_bkps=n_bkps_val)
		change_points_train_stds = detect_change_points(mean_train_stds.values, n_bkps=n_bkps_train)

	# Plot
	inner = GridSpecFromSubplotSpec(2, 1,
					subplot_spec=outer[i], wspace=0.1, hspace=0.33)
	axs = []
	for j in range(2):
		ax = plt.Subplot(fig, inner[j])
		axs.append(ax)
	axs[0].set_xticks([])

	if dataset in ["jasmine", "vehicle"]:
		tmp = val_deriv_stds
		val_deriv_stds = train_deriv_stds
		train_deriv_stds = tmp

	# Validation
	mean_values = np.mean(val_deriv_stds, axis=0)
	max_values = np.max(val_deriv_stds, axis=0)
	min_values = np.min(val_deriv_stds, axis=0)
	variance_values = np.var(val_deriv_stds, axis=0)
	std_deviation = np.sqrt(variance_values)
	windows = np.arange(1, max_iter - window_size * 2 + 2)
	if dataset in ["ImageNet", "Cifar100", "Cifar10"] or "NN" in dataset:
		window_ticks = [1, 10, 20, 30, 40]
		window_ticks_labels = [x * 5 for x in window_ticks]
	elif "Pybnn" in dataset:
		window_ticks = [1, 5, 10]
		window_ticks_labels = [x * 1000 for x in window_ticks]
	else:
		window_ticks = [1, 5, 10, 15]
		window_ticks_labels = [x * 3 for x in window_ticks]

	axs[0].plot(windows[0:], mean_values[0:], label='Mean'
			, color='#0077b6'
			)  # Line plot for the mean
	axs[0].fill_between(windows[0:], 
					 min_values[0:],
					 mean_values[0:] + std_deviation[0:], 
				color='#0077b6', 
				alpha=0.2, label='1 Std Dev')
	axs[0].set_xticks(window_ticks_labels, [])
	axs[0].grid(True)
	delta = mean_values[0:].max() - mean_values[0:].min()
	if "Pybnn" in dataset:
		axs[0].set_ylim(0, mean_values[0:].max() * 0.01)
	else:
		axs[0].set_ylim([mean_values[0:].min() - 0.02 * delta,
				  mean_values[0:].max() + 0.2 * delta])
	
	# Training
	mean_values = np.mean(train_deriv_stds, axis=0)
	max_values = np.max(train_deriv_stds, axis=0)
	min_values = np.min(train_deriv_stds, axis=0)
	variance_values = np.var(train_deriv_stds, axis=0)
	std_deviation = np.sqrt(variance_values)
	axs[1].plot(windows[0:], mean_values[0:], label='Mean'
		, color='#f77f00'
		)  # Line plot for the mean
	axs[1].fill_between(windows[0:], 
					 min_values[0:], 
                    mean_values[0:] + std_deviation[0:],
				color='#f77f00', 
				alpha=0.2, label='1 Std Dev')
	axs[1].set_xticks(window_ticks_labels, window_ticks_labels)
	axs[1].grid(True)
	delta = mean_values[0:].max() - mean_values[0:].min()
	if "Pybnn" in dataset:
		axs[1].set_ylim(0, mean_values[0:].max() * 0.01)
	else:
		axs[1].set_ylim([mean_values[0:].min() - 0.02 * delta,
				  mean_values[0:].max() + 0.2 * delta])

	# stage
	if args.detect_point:
		if dataset in ["Cifar100", "ImageNet"]:
			for p in change_points_val_stds[1:-1]:
				axs[0].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
		else:
			for p in change_points_val_stds[:-1]:
				axs[0].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
		for p in change_points_train_stds[:-1]:
			axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
	else:
		change_point_dict = {
			"Cifar10": {
				"valid": [5, 70, 180],
				"train": [5, 20]
			},
			"Cifar100": {
				"valid": [35, 105, 175],
				"train": [10, 25]
			},
			"ImageNet": {
				"valid": [40, 95, 165],
				"train": [5, 20]
			},
			"Fashion-MNIST": {
				"valid": [5, 15],
				"train": [5, 15],
			},
			"volkert": {
				"valid": [5, 15],
				"train": [5, 15],
			},
			"NN_146606": {
				"valid": [15, 75],
				"train": [15, 75]
			},
			"NN_7592": {
				"valid": [15, 75],
				"train": [15, 75]
			},
			"Pybnn_Boston": {
				"valid": [2385, 3970, 5500],
				"train": [1885, 5000]
			},
			"Pybnn_Protein": {
				"valid": [975, 5000],
				"train": [1885, 5000]
			}
		}
		cpd = change_point_dict[dataset]
		for p in cpd["valid"]:
			axs[0].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
		for p in cpd["train"]:
			axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)

	for j in range(2):
		fig.add_subplot(axs[j])

	if dataset in label_dic.keys():
		axs[0].set_title(label_dic[dataset])
	else:
		axs[0].set_title(dataset)
	
	if int(i%3) == 0:
		axs[0].set_ylabel(" Val.std", fontsize=fontsize)
		axs[1].set_ylabel("Train.std", fontsize=fontsize)
	if int(i/3) == 2:
		axs[1].set_xlabel("Epoch", fontsize=fontsize)
	# exponential y-axis
	if dataset in ["jasmine", "vehicle", "higgs", "volkert", "Pybnn_Boston", "Pybnn_Protein"]:
		for j in range(2):
			axs[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.08)
plt.savefig("std_loss.pdf")
plt.savefig(f"std_loss.png")