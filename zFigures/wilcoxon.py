
import os
import json
import warnings
import argparse
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# from cliffs_delta import cliffs_delta

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--task', type=str, default='acc_loss', help='Description of task')
parser.add_argument('--dataset', type=str, default='ImageNet', help='Path of benchmark data')
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy', 'test_losses'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[20, 50, 81, 120, 150], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=float, default=3, help='Fraction of saving in hyperband')
parser.add_argument('--scheme', type=str, default="BOHB", help='HPO algorithm')
args = parser.parse_args()

label_dic = {'train_accuracy': 'Training accuracy',
             'train_losses': 'Training loss',
             'valid_accuracy': 'Validation accuracy',
             'valid_losses': 'Validation loss'}
title_dic = {'Cifar10': 'CIFAR-10',
             'Cifar100': 'CIFAR-100',
             'ImageNet': 'ImageNet-16-120'}
scheme = args.scheme
max_iters = args.max_iters
ult_objs = args.ult_objs
dataset_path = f'../Records/{scheme}/'
dataset = args.dataset
dataset_path = os.path.join(dataset_path, dataset)

file_name = args.task
eta = args.eta
if eta.is_integer():
    eta = int(eta)

for iter in max_iters:
    print()
    print(f"******** max_iter = {iter} ********")
    for obj in ult_objs:
        print(f"*********** obj = {obj} ************")
        # Load data
        if scheme in ['Hyperband', 'SH']:
            dir = os.path.join(dataset_path, f"Max_iter_{iter}_eta_{eta}", "cta", f"obj_{obj}")
        else:
            dir = os.path.join(dataset_path, f"{scheme}_Max_iter_{iter}_eta_{eta}", "cta", f"obj_{obj}")
        if not os.path.exists(dir):
            warnings.warn(f"Directory {dir} doesn't exist.")
            continue
        
        file = os.path.join(dir, f"{file_name}.csv")
        if not os.path.exists(file):
            print(f"File {file} doesn't exist!!!!!!!!")
            continue
        df = pd.read_csv(file)
        json_rst = dict()
        
        # Pair-wise wilcoxon test
        if not 'train_losses' in df.columns and not 'valid_losses' in df.columns:
            if 'stage_3_loss_seed' in df.columns:
                file = os.path.join(dir, f"loss_seed.csv")
            else:
                file = os.path.join(dir, f"loss.csv")
            df_base = pd.read_csv(file)
            if 'stage_3_win_loss' in df.columns:
                if dataset in ["Cifar10", "Cifar100", "ImageNet"]:
                    file = os.path.join(dir, f"adp_stage_loss_nsb.csv")
                else:
                    file = os.path.join(dir, f"adp_stage_loss_lcb.csv")
                df_base = pd.read_csv(file)
        else:
            df_base = df
        for i, cta1 in enumerate(df_base.columns):
            if cta1 in ('Max_test_accuracy', 'Min_test_loss'):
                continue
            for j in range(len(df.columns)):
                cta2 = df.columns[j]
                print(f"{cta2=}")
                if cta2 in ('Max_test_accuracy', 'Min_test_loss'):
                    continue

                d1 = df_base[cta1]
                d2 = df[cta2]
                if 'seed' in cta1 or 'seed' in cta2:
                    d1 = df_base["Max_test_accuracy"] - d1
                    d2 = df["Max_test_accuracy"] - d2
                d = d1 - d2

                if not (d != 0).any():
                    print("zero")
                    continue

                if 'stage' in cta2 or 'win_mu' in cta2:
                    res = stats.wilcoxon(d, alternative="less")
                    print(f"--> Assume cta1 gets lower test accuracy than cta2, p<0.05 reject null.")
                    print(f"cta1 = {cta1}, cta2 = {cta2}, res.pvalue = {res.pvalue}, max diff = {d.max()}, min diff = {d.min()}, mean diff = {d.mean()}")
                    continue
                elif 'seed' in cta2:
                    res = stats.wilcoxon(d, alternative="greater")
                    print(f"--> Assume d-cta1 gets greater diff to test accuracy than cta2, p<0.05 reject null.")
                    print(f"cta1 = {cta1}, cta2 = {cta2}, res.pvalue = {res.pvalue}, max diff = {d.max()}, min diff = {d.min()}, mean diff = {d.mean()}")
                    continue

                if 'loss' in obj:
                    res = stats.wilcoxon(d, alternative="less")
                    print(f"--> Assume cta1 gets lower test loss than cta2, p<0.05 reject null.")
                else:
                    res = stats.wilcoxon(d, alternative="greater")
                    print(f"--> Assume cta1 gets higher test accuracy than cta2, p<0.05 reject null.")
                # res = stats.wilcoxon(d)
                print(f"cta1 = {cta1}, cta2 = {cta2}, res.pvalue = {res.pvalue}, max diff = {d.max()}, min diff = {d.min()}, mean diff = {d.mean()}")
