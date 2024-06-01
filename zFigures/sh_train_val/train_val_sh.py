import matplotlib.pyplot as plt

import os
import json
import warnings
import argparse
import pandas as pd
# from scipy import stats
# import scikit_posthocs as sp
import seaborn as sns
# import seaborn as sns
import numpy as np

# from cliffs_delta import cliffs_delta

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--task', type=str, default='acc_loss', help='Description of task')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Path of benchmark data')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_losses'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[10, 30, 50, 70, 90, 110, 130, 150], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=int, default=3, help='Fraction of saving in sh')
parser.add_argument('--scheme', type=str, default="SH", help='HPO algorithm')
args = parser.parse_args()
scheme = args.scheme
max_iters = args.max_iters
ult_objs = args.ult_objs
file_name = args.task
eta = args.eta

fontsize = 16
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
# Create the main figure and two subfigures (subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 3.7))  # Two main subfigures in one row
# Remove the original axes as we will add our own sub-subplots
ax1.remove()
ax2.remove()

# Add sub-subfigures in a single row within each main subfigure
n_subsubfigs = 3
widths = [1, 1, 1]  # Equal widths for sub-subfigures

# Create sub-subfigures in the first main subplot area
axs1 = fig.add_gridspec(1, n_subsubfigs, left=0.036, right=0.55, 
						bottom=0.28, top=0.85, wspace=0.18)  # Adjust left, right, and wspace as needed
subfigs1 = []
for i in range(n_subsubfigs):
	subfig1 = fig.add_subplot(axs1[0, i])
	subfig1.set_xlabel("Budget (R)", fontsize=fontsize)
	subfigs1.append(subfig1)
subfigs1[0].set_ylabel("Delta accuracy", fontsize=fontsize)


# Create sub-subfigures in the second main subplot area
axs2 = fig.add_gridspec(1, n_subsubfigs, left=0.6, right=0.993, 
						bottom=0.28, top=0.85, wspace=0.25)  # Adjust left, right, and wspace as needed
subfigs2 = []
for i in range(n_subsubfigs):
	subfig2 = fig.add_subplot(axs2[0, i])
	subfig2.set_xlabel("Fraction of budget", fontsize=fontsize)
	subfigs2.append(subfig2)
subfigs2[0].set_ylabel("Min regret", fontsize=fontsize)



label_dic = {'train_losses': 'Training loss',
			 'valid_losses': 'Validation loss'}

title_dic = {'Cifar10': 'CIFAR-10',
			 'Cifar100': 'CIFAR-100',
			 'ImageNet': 'ImageNet-16-120'}

palette = sns.color_palette("RdBu", n_colors=8)  # You can change "viridis" to any other palette

data_dir = f'../../Records/{scheme}/'
for idx, dataset in enumerate(['Cifar10', 'Cifar100', 'ImageNet']):
	dataset_path = data_dir + dataset
	
	mean_values = dict()
	std_values = dict()
	all_values = dict()
	diff_values = []
	outlier_uvalues = []
	outlier_lvalues = []
	for col in label_dic.keys():
		mean_values[col] = np.zeros(len(max_iters))
		std_values[col] = np.zeros(len(max_iters))
		all_values[col] = []
	for i, iter in enumerate(max_iters):
		if scheme == "Hyperband" or scheme == "SH":
			dir = os.path.join(dataset_path, f"Max_iter_{iter}_eta_{eta}", "cta", f"obj_test_accuracy")
		else:
			dir = os.path.join(dataset_path, f"{scheme}_Max_iter_{iter}_eta_{eta}", "cta", f"obj_test_accuracy")
		file = os.path.join(dir, f"acc_loss.csv")
		df = pd.read_csv(file)
		for col in label_dic.keys():
			mean_values[col][i] = df[col].median()
			std_values[col][i] = df[col].std()
			all_values[col].append(df[col])
		
		diff = (all_values["train_losses"][-1] - all_values["valid_losses"][-1]).values
		diff_values.append(diff)
		
	iters = np.arange(10, 291, 40)
	df = pd.DataFrame(np.array(diff_values).transpose(), columns=[str(x) for x in iters])
	df_melted = df.reset_index().melt(id_vars='index', var_name='max_iters', value_name='values')
	df_melted["max_iters"] = df_melted["max_iters"].astype(int)
	
	# points
	sns.stripplot(
		ax=subfigs1[idx],
		x='max_iters',  # Categories are now the different max_iters
		y='values',  # Values from the melted DataFrame
		hue='max_iters', palette="RdBu",
		data=df_melted, 
		dodge=False, 
		alpha=.1, 
		# zorder=2, 
		jitter=True,
		legend=False, 
	)
	xs = np.arange(len(df_melted['max_iters'].unique())) + 0.25
	positions = np.arange(len(max_iters)) * 2  # Create positions with larger intervals
	
	# violin
	v = subfigs1[idx].violinplot(diff_values, positions=xs, showmeans=False, 
						   showmedians=True, showextrema=True, 
						   widths=1, side="high")
	# Change color of each violin
	for part, color in zip(v['bodies'], palette):
		part.set_facecolor(color)
		part.set_alpha(1)  # Optional: set transparency
		
	# Change the color and linewidth of the extrema lines
	for pc in ['cmaxes', 'cmins', 'cbars', 'cmedians']:
		v[pc].set_edgecolors(palette)	  # Change the color to red
		v[pc].set_color("black")
		v[pc].set_linewidth(1.65)	  # Set the line width to 2

	subfigs1[idx].set_xticks(xs, max_iters)
	subfigs1[idx].grid(linestyle='-.')
	if idx > 0:	
		subfigs1[idx].set_ylabel("")
	subfigs1[idx].set_title(title_dic[dataset], fontsize=fontsize)

for d, dataset in enumerate(['Cifar10', 'Cifar100', 'ImageNet']):
	with open(f'{scheme}_avg_rank_{dataset}.json', 'r') as file:
		mean_rank_dict = json.load(file)
	for criteria in label_dic.keys():
		if dataset in title_dic.keys():
			print(f'{len(mean_rank_dict["150"][criteria]["Min-Regret (acc)"])=}')
			mean_rank_dict["150"][criteria]['Min-Regret (acc)'][0] = 0
			subfigs2[d].plot( mean_rank_dict["150"][criteria]['Min-Regret (acc)'], label=label_dic[criteria], linewidth=2.5)
		else:
			subfigs2[d].plot( mean_rank_dict["30"][criteria]['Min-Regret (acc)'], label=label_dic[criteria], linewidth=2.5)
	if dataset in title_dic.keys():
		subfigs2[d].set_xticks([0, 1.3, 2.6, 4])
		subfigs2[d].set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{-0}$'])
	else:
		subfigs2[d].set_xticks([0, 1, 2])
		subfigs2[d].set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^0$'])

	subfigs2[d].grid(linestyle='-.')
	subfigs2[d].set_title(title_dic[dataset], fontsize=fontsize)

subfigs2[0].legend( bbox_to_anchor=(0.5,1.05), loc=3, ncol = 2, framealpha = 0, fontsize=fontsize)


fig.text(0.26, 0.02, '(a) Distribution of final performance difference', ha='center', fontsize=fontsize+2)
fig.text(0.78, 0.02, '(b) Mean regret-over-time', ha='center', fontsize=fontsize+2)

# Show the plot
plt.savefig('sh_train_val.png')
plt.savefig('sh_train_val.pdf')