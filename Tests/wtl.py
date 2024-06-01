
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
parser.add_argument('--ult_objs', type=str, nargs='+', default=['test_accuracy'], help='List of strings (default: ["test_accuracy", "test_losses"])')
parser.add_argument('--max_iters', type=int, nargs='+', default=[20, 50, 81, 120, 150], help='List of integers (default: [20, 50, 81, 120, 150])')
parser.add_argument('--eta', type=int, default=3, help='Fraction of saving in hyperband')
args = parser.parse_args()

label_dic = {'train_accuracy': 'Training accuracy',
             'train_losses': 'Training loss',
             'valid_accuracy': 'Validation accuracy',
             'valid_losses': 'Validation loss'}
title_dic = {'Cifar10': 'CIFAR-10',
             'Cifar100': 'CIFAR-100',
             'ImageNet': 'ImageNet-16-120'}
max_iters = args.max_iters
ult_objs = args.ult_objs
dataset_path = '../Records/Hyperband/'
dataset = args.dataset
dataset_path = os.path.join(dataset_path, dataset)

if 'Fashion-MNIST' in dataset_path or 'higgs' in dataset_path or 'adult' in dataset_path or 'jasmine' in dataset_path or 'vehicle' in dataset_path or 'volkert' in dataset_path:
    max_iters = [3, 6, 10, 15, 30] 
else:
    max_iters = [20, 50, 81, 120, 150]
file_name = args.task
eta = args.eta

dataset_path = '../Records/Hyperband/'
for d, dataset in enumerate([dataset]):
    print(f"\nDataset = {dataset}")
    for iter in max_iters:
        print(f"**** iter = {iter} ****")
        if dataset in title_dic.keys():
            dir = os.path.join(dataset_path, dataset, f"Max_iter_{iter}_eta_{eta}", "cta", f"obj_test_accuracy")
        else:
            dir = os.path.join(dataset_path, dataset, f"Max_iter_{iter}_eta_{eta}", "cta", f"obj_test_accuracy")
        file = os.path.join(dir, f"{args.task}.csv")
        print(file)
        df = pd.read_csv(file)
        file = os.path.join(dir, "acc_loss.csv")
        df_base = pd.read_csv(file)


        for base in ['train_losses', 'valid_losses', 'train_accuracy', 'valid_accuracy']:
            print(f"\n**** Base = {base} ****")
            wins = dict()
            ties = dict()
            losses = dict()
            maxes= dict()
            means= dict()
            for criteria in df.columns:
                if criteria == base or criteria == 'Max_test_accuracy':
                    continue
                wins[criteria] = (df[criteria] > df_base[base]).sum()
                ties[criteria] = (df[criteria] == df_base[base]).sum()
                losses[criteria] = (df[criteria] < df_base[base]).sum()
                maxes[criteria] = (df_base[base] - df[criteria]).max()
                means[criteria] = (df_base[base] - df[criteria]).mean()
                print(f"criteria = {criteria}, w = {wins[criteria]}, t = {ties[criteria]}, l = {losses[criteria]}")