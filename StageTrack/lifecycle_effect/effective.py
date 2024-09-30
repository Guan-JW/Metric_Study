from tkinter import font
import os, csv, argparse, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--task', type=str, default='loss', help='Description of task')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[20, 50, 81, 120, 150], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=float, default=3, help='Fraction of saving in hyperband')
parser.add_argument('--scheme', type=str, default="Hyperband", help='HPO algorithm')
args = parser.parse_args()


task = args.task

def get_meta(dataset):
    eta = 1.33
    obj = 'test_accuracy'
    if dataset in ["ImageNet", "Cifar10", "Cifar100"]:
        max_iter = 200
        interval = 5
    elif "NN" in dataset:
        max_iter = 243
        interval = 5
    elif "Pybnn" in dataset:
        max_iter = 9700
        interval = 200
        eta = 3
        obj = 'test_loss'
    else:
        max_iter = 50
        interval = 3

    return max_iter, interval, eta, obj

stage_3_start_dict = {
    "ImageNet": 165,
    "Cifar10": 180,
    "Cifar100": 175,
    "Fashion-MNIST": 15,
    "volkert": 15,
    "NN_146606": 75,
    "NN_7592": 75,
    "Pybnn_Boston": 5500
}
stage_2_start_dict = {
    "ImageNet": 95,
    "Cifar10": 70,
    "Cifar100": 95,
    "Pybnn_Boston": 2385
}
label_dic = {"ImageNet": "ImageNet-12-160",
             "Fashion-MNIST": "Fashion-MNIST",
             "NN_7592": "NN-Higgs",
             "Pybnn_Boston": "BNN-Boston"}

fontsize = 12
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 1.5))
# Remove the original axes as we will add our own sub-subplots
ax1.remove()
ax2.remove()

n_subsubfigs = 3
axs1 = fig.add_gridspec(1, n_subsubfigs, left=0.029, right=0.64, 
                        bottom=0.32, top=0.85, wspace=0.2)
subfigs1 = []
for i in range(n_subsubfigs):
    subfig1 = fig.add_subplot(axs1[0, i])
    subfig1.set_xlabel("Budget", fontsize=fontsize)
    subfigs1.append(subfig1)
subfigs1[0].set_ylabel("Accuracy regret  ", fontsize=fontsize)

axs2 = fig.add_gridspec(1, 1, left=0.72, right=0.885, 
                        bottom=0.32, top=0.85)
subfigs2 = fig.add_subplot(axs2[0])
subfigs2.set_ylabel("Loss regret", fontsize=fontsize)
subfigs2.set_xlabel("Budget", fontsize=fontsize)
subfigs2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

for d, dataset in enumerate(["ImageNet", "Fashion-MNIST", "NN_7592", "Pybnn_Boston"]):
    # value_list = range(interval, max_iter, interval)
    max_iter, interval, eta, obj = get_meta(dataset)
    value_list = range(max_iter - interval, 0, -interval)

    if task == "loss":
        mean_array = np.zeros((2, len(value_list)))
        std_array = np.zeros((2, len(value_list)))
        max_array = np.zeros((2, len(value_list)))
        min_array = np.zeros((2, len(value_list)))
    else:
        mean_array = np.zeros(len(value_list))
        std_array = np.zeros(len(value_list))
        max_array = np.zeros(len(value_list))
        min_array = np.zeros(len(value_list))
        likelihood_array = np.zeros(len(value_list))

    no_result_cnt = 0   # budgets that don't have results, skip them
    check_stg2 = False
    max_stg3_esp_cnt = 0
    max_stg1_esp_cnt = 0
    stage_3_start_point = stage_3_start_dict[dataset]
    # points that contain ist esp in stage 3/1
    stg3_points = []
    stg1_points = []
    # check if there's stage 1
    if dataset in stage_2_start_dict.keys():
        check_stg2 = True
        max_stg2_esp_cnt = 0
        stage_2_start_point = stage_2_start_dict[dataset]
        stg2_points = []
    for i, v in enumerate(value_list[::-1]):
        dir_path = os.path.join("/home/SSD/HPO/Empirical_study/Records", args.scheme, dataset)
        file_path = os.path.join(dir_path, f'Max_iter_{v}_eta_{eta}', 'cta', f'obj_{obj}', f'{task}.csv')
        if not os.path.exists(file_path):
            warnings.warn(f"File {file_path} doesn't exist.")
            no_result_cnt += 1
            continue

        df_loss = pd.read_csv(file_path)
        if task == "loss":
            train_diff = df_loss.iloc[:, 0].values - df_loss.iloc[:, 1].values
            valid_diff = df_loss.iloc[:, 0].values - df_loss.iloc[:, 2].values
                
            train_dmean = np.mean(train_diff)
            valid_dmean = np.mean(valid_diff)
            train_dstd = np.std(train_diff)
            valid_dstd = np.std(valid_diff)

            train_dmax = np.max(train_diff)
            valid_dmax = np.max(valid_diff)
            train_dmin = np.min(train_diff)
            valid_dmin = np.min(valid_diff)

            mean_array[0, i] = train_dmean
            mean_array[1, i] = valid_dmean
            std_array[0, i] = train_dstd
            std_array[1, i] = valid_dstd

            max_array[0, i] = train_dmax
            max_array[1, i] = valid_dmax
            min_array[0, i] = train_dmin
            min_array[1, i] = valid_dmin
        else:
            diff = df_loss.iloc[:, 0].values - df_loss.iloc[:, 1].values
            if obj == "test_loss":
                diff *= -1
                zero_cnt = np.sum(diff < np.max(df_loss.iloc[:, 0].values) * 3)
            else:
                zero_cnt = np.sum(diff < np.max(df_loss.iloc[:, 0].values) * 0.015)


            dmean = np.mean(diff)
            dstd = np.std(diff)
            dmax = np.max(diff)
            dmin = np.min(diff)

            mean_array[i] = dmean
            std_array[i] = dstd
            max_array[i] = dmax
            min_array[i] = dmin
            likelihood_array[i] = zero_cnt / diff.shape[0]
        
        # Find the first few budgets that contain increasing es points in the 3rd stage
        # print(f"Total Max = {np.max(df_loss.iloc[:, 0].values)}")
        esp = v
        stg3_esp_cnt = 0
        stg3_esp_list = []
        while (esp > stage_3_start_point):
            esp /= eta
            stg3_esp_list.append(esp)

        if len(stg3_esp_list) > max_stg3_esp_cnt:
            max_stg3_esp_cnt = len(stg3_esp_list)
            stg3_points.append(v)
            print(f"Stage 3 -- {v}, dmean == {mean_array[i]}({(mean_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                      dstd = {std_array[i]}({(std_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                        dmax = {max_array[i]}(({(max_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f})), \
                        dmin = {min_array[i]}({(min_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), max = {np.max(df_loss.iloc[:, 0].values)}")

        # Find the first few budgets that contain increasing es points in the 3rd stage
        if check_stg2:
            stg2_esp_cnt = 0
            stg2_esp_list = []
            # skip points in stage 3
            while (esp > stage_2_start_point and esp < stage_3_start_point):
                esp /= eta
                stg2_esp_list.append(esp)
            if len(stg2_esp_list) > max_stg2_esp_cnt:
                max_stg2_esp_cnt = len(stg2_esp_list)
                stg2_points.append(v)
                print(f"Stage 2 -- {v}, dmean == {mean_array[i]}({(mean_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                            dstd = {std_array[i]}({(std_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                                dmax = {max_array[i]}(({(max_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f})), \
                                dmin = {min_array[i]}({(min_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), max = {np.max(df_loss.iloc[:, 0].values)}")

        # check stage 1
        if v < stage_3_start_point or (check_stg2 and v < stage_2_start_point):
            stg1_esp_cnt = 0
            stg1_esp_list = []
            while (esp > value_list[-1]):
                esp /= eta
                stg1_esp_list.append(esp)
            if len(stg1_esp_list) > max_stg1_esp_cnt:
                max_stg1_esp_cnt = len(stg1_esp_list)
                stg1_points.append(v)
                print(f"Stage 1 -- {v}, dmean == {mean_array[i]}({(mean_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                        dstd = {std_array[i]}({(std_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), \
                            dmax = {max_array[i]}(({(max_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f})), \
                            dmin = {min_array[i]}({(min_array[i] / np.max(df_loss.iloc[:, 0].values)):.5f}), max = {np.max(df_loss.iloc[:, 0].values)}")

    if no_result_cnt == 0:
        no_result_cnt = - max_iter
    value_list = value_list[::-1][: -no_result_cnt]
        
    # specify ax
    if d < 3:
        ax = subfigs1[d]
    else:
        ax = subfigs2

    if task == "loss":
        ax.plot(value_list, mean_array[0, : -no_result_cnt], label = "Train. loss", color='#1f77b4')
        ax.plot(value_list, mean_array[1, : -no_result_cnt], label = "Valid. loss", color='#9467bd')
        ax.fill_between(value_list, 
                            min_array[0, : -no_result_cnt], 
                            mean_array[0, : -no_result_cnt] + std_array[0, : -no_result_cnt], 
                            alpha=0.5, color='#1f77b4')
        ax.fill_between(value_list, 
                            min_array[1, : -no_result_cnt], 
                            mean_array[1, : -no_result_cnt] + std_array[1, : -no_result_cnt], 
                            alpha=0.5, color='#9467bd')
        ax.fill_between(value_list, 
                            min_array[0, : -no_result_cnt], 
                            max_array[0, : -no_result_cnt], 
                            alpha=0.3, color='#1f77b4')
        ax.fill_between(value_list, 
                            min_array[1, : -no_result_cnt], 
                            max_array[1, : -no_result_cnt], 
                            alpha=0.3, color='#9467bd')
        ax.legend()
    else:
        ax.plot(value_list, mean_array[: -no_result_cnt], color='#0077b4', lw=1.7, label="Mean")
        ax.fill_between(value_list, 
                            # mean_array[: -no_result_cnt] - std_array[: -no_result_cnt], 
                            min_array[: -no_result_cnt], 
                            mean_array[: -no_result_cnt] + std_array[: -no_result_cnt], 
                            alpha=0.5, color='#1f77b4', label="1 Std.")
        ax.fill_between(value_list, 
                            min_array[: -no_result_cnt], 
                            max_array[: -no_result_cnt], 
                            alpha=0.2, label="Max.")
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0]/3, 
                    (mean_array[0] + std_array[0]) + (max_array[0] - mean_array[0] - std_array[0]) / 4 )
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        # possibility
        ax_twin = ax.twinx()
        ax_twin.plot(value_list, likelihood_array[: -no_result_cnt], color='red', lw=1.4, label="Prob. best")
        
        if d == 0:
            current_ylim = ax_twin.get_ylim()
            ax_twin.set_ylim(0.42, current_ylim[1]+0.02)
        if d >= 2:
            ax_twin.set_ylabel("Probability", fontsize=fontsize)
            
        # draw vertical lines
        if check_stg2:
            ax.axvline(x=value_list[0], color='purple', linestyle='--', alpha=0.7, lw=1.3, label="Stage 1")
            for p in stg1_points:
                if p < max_iter - no_result_cnt:
                    ax.axvline(x=p, color='purple', linestyle='--', alpha=0.3, lw=1.3)
            ax.axvline(x=stage_2_start_dict[dataset], color='green', linestyle='--', alpha=0.9, lw=1.3, label="Stage 2")
            for p in stg2_points:
                if p < max_iter - no_result_cnt:
                    ax.axvline(x=p, color='green', linestyle='--', alpha=0.3, lw=1.3)
                
        else:
            ax.axvline(x=value_list[0], color='green', linestyle='--', alpha=0.7, lw=1.3, label="Stage 2")
            for p in stg1_points:
                if p < max_iter - no_result_cnt:
                    ax.axvline(x=p, color='green', linestyle='--', alpha=0.3, lw=1.3)
            
        ax.axvline(x=stage_3_start_dict[dataset], color='r', linestyle='--', alpha=0.7, lw=1.3, label="Stage 3")
        for p in stg3_points:
            if (p < max_iter - no_result_cnt):
                ax.axvline(x=p, color='r', linestyle='--', alpha=0.3, lw=1.3)
        
        if d == 0:
            handles, labels = [], []
            for a in [ax_twin, ax]:
                for handle, label in zip(*a.get_legend_handles_labels()):
                    handles.append(handle)
                    labels.append(label)
            fig.legend(handles, labels, loc='center right', ncol=1, 
                                fontsize=10, bbox_to_anchor=(1, 0.5),
                                # bbox_transform=axs[3].transAxes,
                                handlelength=1, labelspacing=0.1,
                                borderpad=0.15)
    ax.set_title(label_dic[dataset], fontsize=fontsize)

plt.savefig(f'effect_{task}.pdf')
plt.savefig(f'effect_{task}.png')
