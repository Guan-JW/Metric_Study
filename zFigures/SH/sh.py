import csv
from statistics import variance
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--store_dir', type=str, default='../../StageTrack/rst/', help='The directory to put results')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name. Candidates are: ImageNet, Cifar100, Cifar10, adult, higgs, jasmine, Fashion-MNIST, vehicle, volkert]')
args = parser.parse_args()

dataset = args.dataset
dir_path = os.path.join(args.store_dir, dataset)

max_iter = 50
if dataset in ["ImageNet", "Cifar100", "Cifar10"]:
    max_iter = 200
val_loss_file = os.path.join(dir_path, "val_loss.csv")

plt.figure(figsize=(5, 1.6))
R = 27
eta = 3
unit = 8
n_configs = [27, 9, 3, 1]
df_val_loss = pd.read_csv(val_loss_file, header=None).iloc[100:127:, :]
df_columns = df_val_loss.columns
for i, nc in enumerate(n_configs[:-1]):
    epoch = int(R / nc)
    print(f"{epoch+1=}")
    # rank according to the next epoch point
    epoch_next = min(epoch * eta * unit, max_iter)
    sorted_df = df_val_loss.sort_values(by=df_columns[epoch_next]) # asending loss
    # plot the droped ones
    nc_next = int(nc / eta)
    for n in range(nc_next, nc):
        plt.plot(sorted_df.iloc[n, 1 : epoch * unit + 1], lw=1.5)
    df_val_loss = sorted_df.iloc[:nc_next, :]

# deal with the last one
assert df_val_loss.shape[0] == 1
plt.plot(df_val_loss.iloc[0, 1 : R * unit + 1], lw=1.5)
 
for pos in n_configs[1:]:
    plt.axvline(x=pos * unit, color='black', linestyle='--', linewidth=0.8)  # Customize color, linestyle, and linewidth

fontsize=10
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Validation loss", fontsize=fontsize)
plt.subplots_adjust(left=0.09, right=0.97, top=0.98, bottom=0.28)
plt.xlim(left=-2, right=203)
plt.savefig("sh.pdf")
plt.savefig("sh.png")