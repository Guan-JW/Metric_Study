import csv
import sys
sys.path.append("../")
sys.path.append("../../")
import json
# from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the color map module
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

import pandas as pd, numpy as np
import os
import argparse


# Moving average function
def moving_average(y, window_size=3):
    return np.convolve(y, np.ones(window_size) / window_size, mode='same')

def generate_decaying_values(length, decay_rate=0.1):
    """
    Generates a list of decaying floating-point numbers between 0 and 1.
    
    Parameters:
    length (int): The size of the list.
    decay_rate (float): The rate at which the values decay (default is 0.1).
    
    Returns:
    list: A list of decaying floating-point numbers.
    """
    # Create an array of indices from 0 to length-1
    x = np.arange(length)
    
    # Apply a linear decay formula scaled between 1 and 0
    decayed_values = np.exp(-decay_rate * x)  # Exponential decay for smoother decay
    
    return decayed_values

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--store_dir', type=str, default='../../StageTrack/rst/', help='The directory to put results')
parser.add_argument('--store_dir_lcb', type=str, default='../../StageTrack/rst_w_test/', help='The directory to put results')
args = parser.parse_args()

datasets = [
	"Cifar10", "Cifar100", "ImageNet", 
	"volkert", "Fashion-MNIST", 
	"NN_146606", "NN_7592",
	"Pybnn_Boston", "Pybnn_Protein"
	]

label_dic = {"Cifar10": "CIFAR-10",
			 "Cifar100": "CIFAR-100",
			 "ImageNet": "ImageNet-16-120",
			 "NN_146606": "NN-Higgs",
			 "NN_7592": "NN-Adult",
			 "Pybnn_Boston": "BNN-Boston",
			 "Pybnn_Protein": "BNN-Protein"}
label_dic_lcb = {
			 "volkert": "Volkert",
			 "Fashion-MNIST": "Fashion-MNIST"}

fig = plt.figure(figsize=(6.5, 6))
outer = GridSpec(3, 3, wspace=0.25, hspace=0.5)
fontsize=11

n_configs = 6
starts = [398, 137, 642]
starts_pybnn = {"Pybnn_Boston": 20,
				"Pybnn_Protein": 197}

for i, dataset in enumerate(datasets):
	max_iter = 50
	if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
		max_iter = 200
	elif "NN" in dataset:
		max_iter = 243
	elif "Pybnn" in dataset:
		max_iter = 10000

	if dataset == "Fashion-MNIST" or dataset == "volkert":
		dir_path = os.path.join(args.store_dir_lcb, dataset)
	else:
		dir_path = os.path.join(args.store_dir, dataset)
	val_loss_file = os.path.join(dir_path, "val_loss.csv")
	train_loss_file = os.path.join(dir_path, "train_loss.csv")
	start = starts[int(i/3)]

	if "Pybnn" in dataset:
		start = starts_pybnn[dataset]
	df_val_loss = pd.read_csv(val_loss_file, header=None).iloc[start:start+n_configs, 1:]
	df_train_loss = pd.read_csv(train_loss_file, header=None).iloc[start:start+n_configs, 1:]

	inner = GridSpecFromSubplotSpec(2, 1,
					subplot_spec=outer[i], wspace=0.1, hspace=0.3)
	axs = []
	for j in range(2):
		ax = plt.Subplot(fig, inner[j])
		axs.append(ax)
	axs[0].set_xticks([])
	
	for nc in range(n_configs):
		train_loss = df_train_loss.iloc[nc,:].values
		val_loss = df_val_loss.iloc[nc,:].values
		axs[0].plot(val_loss, lw=1.2)
		axs[1].plot(train_loss, lw=1.2)

	for j in range(2):
		fig.add_subplot(axs[j])

	if dataset in label_dic.keys():
		axs[0].set_title(label_dic[dataset])
		axs[0].set_xlim(-5, max_iter+5)
		axs[1].set_xlim(-5, max_iter+5)
	elif dataset in label_dic_lcb.keys():
		axs[0].set_title(label_dic_lcb[dataset])
		axs[0].set_xlim(-1, max_iter)
		axs[1].set_xlim(-1, max_iter)
	else:
		axs[0].set_title(dataset)
		axs[0].set_xlim(-1, max_iter)
		axs[1].set_xlim(-1, max_iter)
	
	if dataset == "Pybnn_Boston":
		axs[0].set_ylim(380, 1500)
		axs[1].set_ylim(580, 1500)
	elif dataset == "Pybnn_Protein":
		# axs[0].set_ylim(80, 130)
		axs[1].set_ylim(80, 120)
	elif dataset == "NN_7592":
		axs[0].set_ylim(0.3, 1.2)
		axs[1].set_ylim(0.3, 1.2)
	
	if int(i%3) == 0:
		axs[0].set_ylabel("  Val.loss", fontsize=fontsize)
		axs[1].set_ylabel("Train.loss", fontsize=fontsize)
	if int(i/3) == 2:
		axs[1].set_xlabel("Epoch", fontsize=fontsize)

	if dataset in ["Pybnn_Boston"]:
		for j in range(2):
			axs[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			axs[j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	elif dataset in ["Pybnn_Protein"]:
		for j in range(2):
			axs[j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08)
plt.savefig("overview.pdf")
plt.savefig("overview.png")