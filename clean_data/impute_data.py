import pandas as pd
import numpy as np
import os
import sys
import miceforest as mf
import statistics

def visualize_results(kernelMeanMatch):
    print(kernelMeanMatch)
    print("numerical vars:")
    print(kernelMeanMatch._get_num_vars())
    print("categorical vars:")
    print(kernelMeanMatch._get_cat_vars())
    print(kernelMeanMatch.plot_imputed_distributions(wspace=0.5, hspace=0.7))
    print(kernelMeanMatch.plot_correlations())
    print(kernelMeanMatch.plot_mean_convergence(wspace=0.5, hspace=0.7))
    print(kernelMeanMatch.plot_feature_importance(annot=True, cmap="Blues", vmin=0, vmax=1))

def save_data(kernelMeanMatch, df_stats):
    new_data = kernelMeanMatch.complete_data(0)
    new_data.reset_index(drop=True, inplace=True)
    df_stats.reset_index(drop=True, inplace=True)
    new_data = pd.concat([new_data, df_stats], axis=1)
    new_data.to_csv('../output/out1.csv')

    new_data1 = kernelMeanMatch.complete_data(1)
    new_data1.reset_index(drop=True, inplace=True)
    new_data1 = pd.concat([new_data1, df_stats], axis=1)
    new_data1.to_csv('../output/out2.csv')
    
    new_data2 = kernelMeanMatch.complete_data(2)
    new_data2.reset_index(drop=True, inplace=True)
    new_data2 = pd.concat([new_data2, df_stats], axis=1)
    new_data2.to_csv('../output/out3.csv')
    
    new_data3 = kernelMeanMatch.complete_data(3)
    new_data3.reset_index(drop=True, inplace=True)
    new_data3 = pd.concat([new_data3, df_stats], axis=1)
    new_data3.to_csv('../output/out4.csv')
    
    new_data4 = kernelMeanMatch.complete_data(4)
    new_data4.reset_index(drop=True, inplace=True)
    new_data4 = pd.concat([new_data4, df_stats], axis=1)
    new_data4.to_csv('../output/out5.csv')
    return new_data

def imputation(new_data, df_stats):
    kernelMeanMatch = mf.MultipleImputedKernel(new_data, datasets=5, save_all_iterations=True, random_state=42, mean_match_candidates=5)
    kernelMeanMatch.mice(5)
    visualize_results(kernelMeanMatch)
    new_data = save_data(kernelMeanMatch, df_stats)
    return new_data

def impute_data(df_clean):
    df_clean = df_clean[df_clean.loc[:, ['PERCBESTINKO', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA', 'WOZ']].isna().sum(axis=1) < 2]
    with open('../output/NaNs.txt', 'a') as f:
        print("", file=f)
        print("All NaNs before imputation:", file=f)
        print(df_clean.isna().sum(), file=f)
        print(df_clean.shape, file=f)
    df_clean.dropna(subset = ['WOZ', 'PERCBESTINKO', 'SECMMA', 'SECMPA'], inplace=True)
    with open('../output/NaNs.txt', 'a') as f:
        print("", file=f)
        print("All NaNs when only impute education parents:", file=f)
        print(df_clean.isna().sum(), file=f)
        print(df_clean.shape, file=f)
    print("age statistics:")
    print(df_clean.shape)
    print(df_clean['age'].mean() / 365.25)
    print(statistics.stdev(df_clean['age']) / 365.25)
    df_stats = df_clean[['age']].copy()
    df_stats.to_csv('../output/df_stats.csv')
    df_clean = df_clean[['GENDER', 'ETHN', 'WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'LEVEL3', 'WOZ', 'PERCBESTINKO', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA']].copy()
    df_clean = df_clean.astype({'OPLNIVMA': 'category', 'OPLNIVPA' : 'category', 'SECMMA': 'category', 'SECMPA': 'category'})
    df_clean.to_csv('../output/out.csv')
    # either impute in this file, or use imput.ipynb file
    df_clean = imputation(df_clean, df_stats)
    return df_clean
