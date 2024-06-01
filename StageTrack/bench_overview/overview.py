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

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--store_dir', type=str, default='../../StageTrack/rst_w_test/', help='The directory to put results')
args = parser.parse_args()

datasets = ["Cifar10", "Cifar100", "ImageNet", 
	"Fashion-MNIST", "jasmine", "vehicle", 
	"adult", "higgs", "volkert"]

label_dic = {"Cifar10": "CIFAR-10",
			 "Cifar100": "CIFAR-100",
			 "ImageNet": "ImageNet-16-120"}

fig = plt.figure(figsize=(6.5, 6))
outer = GridSpec(3, 3, wspace=0.25, hspace=0.5)

n_configs = 6
start = 672
starts = [398, 135, 137]
fontsize=11
for i, dataset in enumerate(datasets):
	max_iter = 50
	if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
		max_iter = 200

	dir_path = os.path.join(args.store_dir, dataset)
	val_loss_file = os.path.join(dir_path, "val_loss.csv")
	train_loss_file = os.path.join(dir_path, "train_loss.csv")
	start = starts[int(i/3)]
	df_val_loss = pd.read_csv(val_loss_file, header=None).iloc[start:start+n_configs, 1:]
	df_train_loss = pd.read_csv(train_loss_file, header=None).iloc[start:start+n_configs, 1:]

	inner = GridSpecFromSubplotSpec(2, 1,
					subplot_spec=outer[i], wspace=0.1, hspace=0.1)
	axs = []
	for j in range(2):
		ax = plt.Subplot(fig, inner[j])
		axs.append(ax)
	axs[0].set_xticks([])
	
	for nc in range(n_configs):
		if dataset in ["jasmine", "vehicle"]:
			axs[0].plot(df_train_loss.iloc[nc,:].values, lw=1.2)
			axs[1].plot(df_val_loss.iloc[nc,:].values, lw=1.2)
		else:
			axs[0].plot(df_val_loss.iloc[nc,:].values, lw=1.2)
			axs[1].plot(df_train_loss.iloc[nc,:].values, lw=1.2)

	for j in range(2):
		fig.add_subplot(axs[j])

	if dataset in label_dic.keys():
		axs[0].set_title(label_dic[dataset])
		axs[0].set_xlim(-5, max_iter+5)
		axs[1].set_xlim(-5, max_iter+5)
	else:
		axs[0].set_title(dataset)
		axs[0].set_xlim(-1, max_iter)
		axs[1].set_xlim(-1, max_iter)
	
	if int(i%3) == 0:
		axs[0].set_ylabel("  Val.loss", fontsize=fontsize)
		axs[1].set_ylabel("Train.loss", fontsize=fontsize)
	if int(i/3) == 2:
		axs[1].set_xlabel("Epoch", fontsize=fontsize)


plt.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08)
plt.savefig("overview.pdf")
plt.savefig("overview.png")