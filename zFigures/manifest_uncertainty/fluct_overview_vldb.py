import sys, os
sys.path.append("../../")
import argparse
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the color map module
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Rounds of running')
args = parser.parse_args()


dataset = args.dataset

fig, axs = plt.subplots(2, 1, figsize=(6,3.2))

plt.rcParams["legend.markerscale"] = 3


file_name = f'save_dict_{dataset}.json'
with open('save_dict_ImageNet.json', 'r') as file:
    save_dict_image = json.load(file)
with open('save_dict_Cifar10.json', 'r') as cifar10_file:
    save_dict_cifar10 = json.load(cifar10_file)
with open('save_dict_Cifar100.json', 'r') as cifar100_file:
    save_dict_cifar100 = json.load(cifar100_file)

num_lines = 20
color_palette = cm.get_cmap('tab20', num_lines)
line_colors = color_palette(range(num_lines))

winsize = 5
point_num = 800
datasets = ['ImageNet', 'Cifar10', 'Cifar100']
markers = ['o', 'v', '*']
dir_path = '/home/SSD/HPO/Empirical_study/StageTrack/rst'
for i, d in enumerate(datasets):
    df_test_loss = pd.read_csv(os.path.join(dir_path, d, 'test_loss.csv'))
    df_valid_loss = pd.read_csv(os.path.join(dir_path, d, 'val_loss.csv'))

    final_valid_loss = np.zeros(point_num)
    valid_fluctuation = np.zeros(point_num)
    for i in range(point_num):
        s = i + 10
        final_valid_loss[i] = df_test_loss.iloc[s, -1]
        valid_fluctuation[i] = np.std(df_valid_loss.iloc[s, 1: 1+10].values)

axins1 = axs[1].inset_axes((0.68, 0.38, 0.3, 0.6))
axins2 = axs[0].inset_axes((0.68, 0.38, 0.3, 0.6))

save_dict = save_dict_image
for i, s in enumerate([2, 5, 6]):

    train_losses_777 = np.array(save_dict[str(s)]['train_losses_777'])
    train_losses_888 = np.array(save_dict[str(s)]['train_losses_888'])
    train_losses_999 = np.array(save_dict[str(s)]['train_losses_999'])
    valid_losses_777 = np.array(save_dict[str(s)]['valid_losses_777'])
    valid_losses_888 = np.array(save_dict[str(s)]['valid_losses_888'])
    valid_losses_999 = np.array(save_dict[str(s)]['valid_losses_999'])

    axs[1].plot(valid_losses_777, color=line_colors[i*2], lw=1)

    valid_loss_mean = (np.array(valid_losses_777) + np.array(valid_losses_888) + np.array(valid_losses_999)) / 3
    train_loss_mean = (np.array(train_losses_777) + np.array(train_losses_888) + np.array(train_losses_999)) / 3
    
    valid_loss_up = [max(valid_losses_777[i], valid_losses_888[i], valid_losses_999[i]) for i in range(len(valid_losses_777))]
    valid_loss_lo = [min(valid_losses_777[i], valid_losses_888[i], valid_losses_999[i]) for i in range(len(valid_losses_777))]
    train_loss_up = [max(train_losses_777[i], train_losses_888[i], train_losses_999[i]) for i in range(len(train_losses_777))]
    train_loss_lo = [min(train_losses_777[i], train_losses_888[i], train_losses_999[i]) for i in range(len(train_losses_777))]
    axs[0].plot(valid_loss_mean, color=line_colors[i*2], lw=1)
    axs[0].fill_between(np.arange(0, 200), valid_loss_lo, valid_loss_up, color=line_colors[i*2], alpha=0.3)

    if i == 0:
        zone_left = 45
        zone_right = 55

        x_ratio = 0.5 
        y_ratio = 0.5 

        x = list(range(200))
        xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
        xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

        y = np.hstack((valid_losses_777[zone_left:zone_right], valid_losses_777[zone_left:zone_right], valid_losses_777[zone_left:zone_right]))
        ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
        ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

        y = np.hstack((valid_loss_up[zone_left:zone_right], valid_loss_up[zone_left:zone_right], valid_loss_up[zone_left:zone_right]))
        ylim2 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
        ylim3 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
    
    axins1.plot(valid_losses_777, color=line_colors[i*2], lw=1)
    axins2.plot(valid_loss_mean, color=line_colors[i*2], lw=1)
    axins2.fill_between(np.arange(0, 200), valid_loss_lo, valid_loss_up, color=line_colors[i*2], alpha=0.3)

fontsize=12
axs[1].set_ylabel('Validation \n loss', fontsize=fontsize)
axs[1].set_xlabel('Epoch')
axs[0].set_ylabel('Validation \n loss', fontsize=fontsize)
axs[0].set_xlabel('Epoch')

axins1.set_xlim(xlim0, xlim1)
axins1.set_ylim(ylim0, ylim1)

axins2.set_xlim(xlim0, xlim1)
axins2.set_ylim(ylim0, ylim1)
axins2.set_xticks([])
axins1.set_xticks([])
axins2.set_yticks([])
axins1.set_yticks([])
mark_inset(axs[1], axins1, loc1=2, loc2=4, fc="none", ec='k', lw=1, alpha=0.7)
mark_inset(axs[0], axins2, loc1=2, loc2=4, fc="none", ec='k', lw=1, alpha=0.7)

axs[1].set_title('(b) Validation loss curve', y=-0.8)
axs[0].set_title('(a) Losses across random seeds', y=-0.8)

plt.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.2, hspace=0.8)
plt.savefig(f'fluct_overview_{dataset}_vldb.png')
plt.savefig(f'fluct_overview_{dataset}_vldb.pdf')
