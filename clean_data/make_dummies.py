import pandas as pd
import numpy as np
import os

pd.set_option('max_columns', 100)
pd.set_option('max_rows', 20)
cwd = os.getcwd()

datasets = ['out1', 'out2', 'out3', 'out4', 'out5']
for set in datasets:
    df = pd.read_csv(cwd + "/../output/" + set + ".csv", index_col=0)
    df = pd.get_dummies(data=df, drop_first=False, columns=['ETHN', 'LEVEL3'])
    df = pd.get_dummies(data=df, drop_first=True, columns=['WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA'])

    # drop age columns
    df = df.drop(columns=['YYYY', 'MM', 'DD'])

    # save dataset with dummies
    df.to_csv("../output/" + set + "_dum.csv")
