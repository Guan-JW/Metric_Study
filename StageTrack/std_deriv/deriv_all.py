import csv
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
parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name. Candidates are: ImageNet, Cifar100, Cifar10, adult, higgs, jasmine, Fashion-MNIST, vehicle, volkert]')
parser.add_argument('--cal_deriv', type=bool, default=False, help='If we should calculate derivative or not')
args = parser.parse_args()

datasets = ["Cifar10", "Cifar100", "ImageNet", 
	"Fashion-MNIST", "jasmine", "vehicle", 
	"adult", "higgs", "volkert"]

label_dic = {"Cifar10": "CIFAR-10",
			 "Cifar100": "CIFAR-100",
			 "ImageNet": "ImageNet-16-120"}

fig = plt.figure(figsize=(6.5, 6))
outer = GridSpec(3, 3, wspace=0.35, hspace=0.5)

fontsize=11
for i, dataset in enumerate(datasets):
	max_iter = 50
	window_size = 3 # threshold
	dir_path = os.path.join(args.store_dir_lcb, dataset)
	n_bkps_val = 2
	n_bkps_train = 2
	if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
		max_iter = 200
		window_size = 5
		dir_path = os.path.join(args.store_dir_nb201, dataset)
		n_bkps_train = 3
	if dataset in ["Cifar10"]:
		n_bkps_val = 10
		n_bkps_train = 10
		
	# files that record the derivatives of losses for each configuration at each epoch
	val_loss_deriv_file = os.path.join(dir_path, "val_loss_deriv.csv")
	train_loss_deriv_file = os.path.join(dir_path, "train_loss_deriv.csv")
	df_val_loss_deriv = pd.read_csv(val_loss_deriv_file, header=None)
	df_train_loss_deriv = pd.read_csv(train_loss_deriv_file, header=None)
	config_num_val = df_val_loss_deriv.shape[0]
	config_num_train = df_train_loss_deriv.shape[0]
	# check iteration number 
	assert config_num_val == config_num_train

	config_num = config_num_val
	window_cnt = round((max_iter - 1) / window_size)
	# remove the first column of the dfs, because they are configuration names / ids

	if df_val_loss_deriv.shape[1] == max_iter:
		df_val_loss_deriv = df_val_loss_deriv.iloc[:, 1:]
		df_train_loss_deriv = df_train_loss_deriv.iloc[:, 1:]

	def detect_change_points(signal, penalty=3, model="linear", n_bkps=2):
		if model == "linear":
			algo = rpt.Dynp(model="clinear").fit(signal)
		elif model == "rbf":
			algo = rpt.Dynp(model="rbf").fit(signal)
		result = algo.predict(n_bkps=n_bkps)
		return result
	
	mean_val_deriv = df_val_loss_deriv.mean()
	mean_train_deriv = df_train_loss_deriv.mean()
	# deriv
	change_points_val_deriv = detect_change_points(mean_val_deriv.values, model="rbf", n_bkps=n_bkps_val)
	change_points_train_deriv = detect_change_points(mean_train_deriv.values, model="rbf", n_bkps=n_bkps_train)
	print(f"{dataset=}, {change_points_val_deriv=}")
	print(f"{dataset=}, {change_points_train_deriv=}")
	# Plot
	inner = GridSpecFromSubplotSpec(2, 1,
					subplot_spec=outer[i], wspace=0.1, hspace=0.33)
	axs = []
	for j in range(2):
		ax = plt.Subplot(fig, inner[j])
		axs.append(ax)
	axs[0].set_xticks([])
	
	epochs = np.arange(1, max_iter)
	if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
		epoch_ticks = [1, 50, 100, 150, 200]
	else:
		epoch_ticks = [1, 25, 50]
		
	if dataset == "vehicle":
		tmp = df_train_loss_deriv
		df_train_loss_deriv = df_val_loss_deriv
		df_val_loss_deriv = tmp
	# Validation
	val_loss_deriv_mean = df_val_loss_deriv.mean().values
	val_loss_deriv_std = df_val_loss_deriv.std().values
	val_loss_deriv_max_values = df_val_loss_deriv.max().values
	val_loss_deriv_min_values = df_val_loss_deriv.min().values
	axs[0].plot(epochs, val_loss_deriv_mean, label='Mean'
			, color='#0077b6'
			)  # Line plot for the mean
	axs[0].fill_between(epochs, 
					 val_loss_deriv_mean - val_loss_deriv_std, 
					 val_loss_deriv_mean + val_loss_deriv_std, 
				color='#0077b6', 
				alpha=0.2, label='1 Std Dev')
	axs[0].set_xticks(epoch_ticks, [])
	axs[0].grid(True)
	delta = val_loss_deriv_mean.max() - val_loss_deriv_mean.min()
	axs[0].set_ylim([val_loss_deriv_mean.min() - 0.2 * delta,
				  	 val_loss_deriv_mean.max() + 0.2 * delta])
	
	# Training
	train_loss_deriv_mean = df_train_loss_deriv.mean().values
	train_loss_deriv_std = df_train_loss_deriv.std().values
	train_loss_deriv_max_values = df_train_loss_deriv.max().values
	train_loss_deriv_min_values = df_train_loss_deriv.min().values
	axs[1].plot(epochs, train_loss_deriv_mean, label='Mean'
		, color='#f77f00'
		)  # Line plot for the mean
	axs[1].fill_between(epochs, 
					 train_loss_deriv_mean - train_loss_deriv_std, 
					 train_loss_deriv_mean + train_loss_deriv_std, 
				color='#f77f00', 
				alpha=0.2, label='1 Std Dev')
	axs[1].set_xticks(epoch_ticks, epoch_ticks)
	axs[1].grid(True)
	delta = train_loss_deriv_mean.max() - train_loss_deriv_mean.min()
	axs[1].set_ylim([train_loss_deriv_mean.min() - 0.2 * delta,
				  	 train_loss_deriv_mean.max() + 0.2 * delta])
	
	# stage points
	for p in change_points_val_deriv[-3:-1]:
		axs[0].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
	if dataset == "Cifar10":
		for point in [4, 6]:
			p = change_points_train_deriv[point]
			axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
	elif dataset in ["Cifar100", "ImageNet"]:
		for p in change_points_train_deriv[:-2]:
			axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)
	else:
		for p in change_points_train_deriv[:-1]:
			axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.7, lw=1.2)

	# subgraph adjusting
	for j in range(2):
		fig.add_subplot(axs[j])

	if dataset in label_dic.keys():
		axs[0].set_title(label_dic[dataset])
	else:
		axs[0].set_title(dataset)

	if int(i%3) == 0:
		axs[0].set_ylabel("dVal", fontsize=fontsize)
		axs[1].set_ylabel("dTrain", fontsize=fontsize)
	if int(i/3) == 2:
		axs[1].set_xlabel("Epoch", fontsize=fontsize)
	# # exponential y-axis
	if dataset in ["vehicle", "higgs"]:
		for j in range(2):
			axs[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.08)
plt.savefig("deriv.pdf")
plt.savefig("deriv.png")