import sys, os, re
sys.path.append("../../")
import argparse
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the color map module
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ConfigSpace.configuration_space import Configuration
from matplotlib.gridspec import GridSpecFromSubplotSpec

def cal_param_count(config_dict):
    param_count = 0
    for key in config_dict.keys():
        edge = config_dict[key]
        if edge == 'nor_conv_1x1':
            param_count += 1 * 1 * (16 * 16 + 32 * 32 + 64 * 64) * 5
        elif edge == 'nor_conv_3x3':
            param_count += 3 * 3 * (16 * 16 + 32 * 32 + 64 * 64) * 5
    return param_count

def parse_config_cal_param(cs, config_string):
    # Define regex pattern to capture each connection and value
    pattern = r"(\d+)<-(\d+), Value: '([\w_]+)'"
    
    # Find all matches in the configuration string
    matches = re.findall(pattern, config_string)
    
    # Create a dictionary to hold the configuration
    config_dict = {}
    # Process matches and build configuration dictionary
    for match in matches:
        to_node, from_node, value = match
        config_dict[f'{to_node}<-{from_node}'] = value
    # print(f"{config_dict=}")
    param_count = cal_param_count(config_dict)
    
    # Create a Configuration object from the dictionary
    configuration = Configuration(cs, values=config_dict)

    return configuration, param_count

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Rounds of running')
args = parser.parse_args()
dataset = args.dataset

fig, subaxs = plt.subplots(1, 3, figsize=(6, 1.6), gridspec_kw={'wspace': 0.5})
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
fontsize = 12

winsize = 5
point_num = 800
datasets = ['ImageNet', 'Cifar10', 'Cifar100']
markers = ['o', 'v', '*']
dir_path = '/home/SSD/HPO/Empirical_study/StageTrack/rst'

for iter, d in enumerate(datasets):

    # load configuration space
    if dataset == "ImageNet":
        from DataLoader.test_nasbench_201 import ImageNetNasBench201Benchmark
        print("here")
        Bench = ImageNetNasBench201Benchmark(rng=1)
        print("here")
    elif dataset == 'Cifar100':
        from DataLoader.test_nasbench_201 import Cifar100NasBench201Benchmark
        Bench = Cifar100NasBench201Benchmark(rng=1)
    elif dataset == 'Cifar10':
        from DataLoader.test_nasbench_201 import Cifar10ValidNasBench201Benchmark
        Bench = Cifar10ValidNasBench201Benchmark(rng=1)
    cs = Bench.get_configuration_space(seed=1)

    # load loss data
    df_test_loss = pd.read_csv(os.path.join(dir_path, d, 'test_loss.csv'))
    df_valid_loss = pd.read_csv(os.path.join(dir_path, d, 'val_loss.csv'))

    final_valid_loss = np.zeros(point_num)
    valid_fluctuation = np.zeros(point_num)
    param_counts = np.zeros(point_num)
    param_loss = dict()
    param_fluct = dict()
    for i in range(point_num):
        s = i + 10

        # get configuration
        config_str = df_valid_loss.iloc[s, 0]  # Note: must use the config string from train or validation loss .csv files!
        _, param_count = parse_config_cal_param(cs, config_str)
        param_counts[i] = param_count
        
        # get early-stage fluctuation
        final_valid_loss[i] = df_test_loss.iloc[s, -1]
        valid_fluctuation[i] = np.std(df_valid_loss.iloc[s, 1: 1+10].values)

        # record results
        if param_count in param_loss.keys():
            param_loss[param_count].append(final_valid_loss[i])
            param_fluct[param_count].append(valid_fluctuation[i])
        else:
            param_loss[param_count] = [final_valid_loss[i]]
            param_fluct[param_count] = [valid_fluctuation[i]]
        
    # sort 
    param_loss = {key: param_loss[key] for key in sorted(param_loss)}
    param_fluct = {key: param_fluct[key] for key in sorted(param_fluct)}
    mean_losses = np.zeros(len(param_loss))
    std_losses = np.zeros(len(param_loss))
    mean_fluct = np.zeros(len(param_fluct))
    std_fluct = np.zeros(len(param_fluct))
    for n, key in enumerate(param_loss.keys()):
        mean_losses[n] = (np.mean(np.array(param_loss[key])))
        std_losses[n] = (np.std(np.array(param_loss[key])))
        mean_fluct[n] = (np.mean(np.array(param_fluct[key])))
        std_fluct[n] = (np.std(np.array(param_fluct[key])))
        
    subaxs[iter].plot(list(param_fluct.keys())[2:], mean_fluct[2:], 
                      label=f'Val. loss fluctuation',
                      color='#1f77b4')
    subaxs[iter].fill_between(list(param_fluct.keys())[2:], 
                             mean_fluct[2:] - std_fluct[2:], 
                             mean_fluct[2:] + std_fluct[2:], 
                             alpha=0.3, color='#1f77b4')
    subaxs[iter].set_xticks([0, 5e5, 1e6])
    subaxs[iter].set_xlabel('# Param.')
    if iter == 0:
        subaxs[iter].set_title(f'   {d}', fontsize=10, y=0.98)
    else:
        subaxs[iter].set_title(f' {d}', fontsize=10, y=0.98)
    subaxs[iter].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    subaxs2 = subaxs[iter].twinx()
    subaxs2.plot(list(param_loss.keys())[2:], mean_losses[2:], label=f'Final test loss', color='#9467bd')
    subaxs2.fill_between(list(param_loss.keys())[2:], 
                             mean_losses[2:] - std_losses[2:], 
                             mean_losses[2:] + std_losses[2:], 
                             alpha=0.3, color='#9467bd')
    subaxs2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    if iter == 0:
        # Add legend
        handles, labels = [], []
        for ax in [subaxs[iter], subaxs2]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
            labels.append(label)
        fig.legend(handles, labels, loc='upper center', ncol=2, 
                fontsize=10, bbox_to_anchor=(0.5, 1.03),
                # bbox_transform=axs[3].transAxes,
                handlelength=0.6, labelspacing=0, 
                borderpad=0.25)
    
    if iter == len(datasets) - 1:
        subaxs2.set_ylabel("Final test loss", fontsize=fontsize)

subaxs[0].set_ylabel("Loss Std.", fontsize=fontsize)
plt.subplots_adjust(left=0.07, right=0.935, top=0.72, bottom=0.26)
plt.savefig(f'capacity.png')
plt.savefig(f'capacity.pdf')
