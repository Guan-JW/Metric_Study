import csv
from statistics import variance
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--store_dir', type=str, default='./rst_w_test/', help='The directory to put results')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name. Candidates are: ImageNet, Cifar100, Cifar10, adult, higgs, jasmine, Fashion-MNIST, vehicle, volkert]')
args = parser.parse_args()

dataset = args.dataset
dir_path = os.path.join(args.store_dir, dataset)

max_iter = 50
window_size = 3 # threshold
if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
	max_iter = 200
	window_size = 5

# files that record the derivatives of losses for each configuration at each epoch
val_loss_deriv_file = os.path.join(dir_path, "val_loss_deriv.csv")
train_loss_deriv_file = os.path.join(dir_path, "train_loss_deriv.csv")
df_val_loss_deriv = pd.read_csv(val_loss_deriv_file, header=None)
df_train_loss_deriv = pd.read_csv(train_loss_deriv_file, header=None)
config_num_val = df_val_loss_deriv.shape[0]
config_num_train = df_train_loss_deriv.shape[0]

# check iteration number 
assert max_iter == df_val_loss_deriv.shape[1]
assert max_iter == df_train_loss_deriv.shape[1]
assert config_num_val == config_num_train

config_num = config_num_val
window_cnt = round((max_iter - 1) / window_size)
# remove the first column of the dfs, because they are configuration names / ids
df_val_loss_deriv = df_val_loss_deriv.iloc[:, 1:]
df_train_loss_deriv = df_train_loss_deriv.iloc[:, 1:]
# Initialization, use to store stds for each configuration
val_deriv_stds = np.zeros((config_num, window_cnt))
train_deriv_stds = np.zeros((config_num, window_cnt))
# Loop over to calculate stds.
for c in range(config_num):
	for i in range(window_cnt):
		val_deriv_stds[c, i] = np.std(df_val_loss_deriv.iloc[c, i * window_size: (i+1) * window_size])
		train_deriv_stds[c, i] = np.std(df_train_loss_deriv.iloc[c, i * window_size: (i+1) * window_size])

# Plot
# Validation
plt.figure(figsize=(10, 6))
plt.boxplot(val_deriv_stds, notch=True, patch_artist=True)


plt.xlabel('Period')
plt.ylabel('Std of validation loss derivatives')
plt.grid(True)
pic_file = os.path.join(dir_path, f"valid_loss_deriv_std_window_{window_size}.png")
plt.savefig(pic_file)


# Training
plt.figure(figsize=(10, 6))
plt.boxplot(train_deriv_stds, notch=True, patch_artist=True)
plt.xlabel('Period')
plt.ylabel('Std of training loss derivatives')
plt.grid(True)
pic_file = os.path.join(dir_path, f"train_loss_deriv_std_window_{window_size}.png")
plt.savefig(pic_file)

# mean-variance plot
# Validation
mean_values = np.mean(val_deriv_stds, axis=0)
variance_values = np.var(val_deriv_stds, axis=0)
std_deviation = np.sqrt(variance_values)
windows = np.arange(1, window_cnt + 1)
plt.figure(figsize=(10, 6))
plt.plot(windows, mean_values, label='Mean', color='blue')  # Line plot for the mean
plt.fill_between(windows, mean_values - std_deviation, mean_values + std_deviation, color='blue', alpha=0.3, label='1 Std Dev')
plt.xlabel('Period')
plt.ylabel('Std of validation loss derivatives')
plt.grid(True)
plt.xticks(windows)  # Ensure all window labels are shown
pic_file = os.path.join(dir_path, f"valid_loss_deriv_std_mean_window_{window_size}.png")
plt.savefig(pic_file)
# Training
mean_values = np.mean(train_deriv_stds, axis=0)
variance_values = np.var(train_deriv_stds, axis=0)
std_deviation = np.sqrt(variance_values)
plt.figure(figsize=(10, 6))
plt.plot(windows, mean_values, label='Mean', color='blue')  # Line plot for the mean
plt.fill_between(windows, mean_values - std_deviation, mean_values + std_deviation, color='blue', alpha=0.3, label='1 Std Dev')
plt.xlabel('Period')
plt.ylabel('Std of training loss derivatives')
plt.grid(True)
plt.xticks(windows)  # Ensure all window labels are shown
pic_file = os.path.join(dir_path, f"train_loss_deriv_std_mean_window_{window_size}.png")
plt.savefig(pic_file)