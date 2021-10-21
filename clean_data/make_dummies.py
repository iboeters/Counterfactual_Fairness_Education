import pandas as pd
import numpy as np
import os
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 20)
cwd = os.getcwd()

df = pd.read_csv(cwd + "/../output/out1.csv", index_col=0)
df = pd.get_dummies(data=df, drop_first=False, columns=['ETHN', 'LEVEL3'])
df = pd.get_dummies(data=df, drop_first=True, columns=['WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA'])

# drop age columns
df = df.drop(columns=['YYYY', 'MM', 'DD'])

# save dataset with dummies
df.to_csv('../output/out1_dum.csv')
