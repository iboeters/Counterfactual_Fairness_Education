import pandas as pd
import numpy as np
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from CEVAE import CEVAE_model
from argparse import ArgumentParser

cwd = os.getcwd()
n_con = 2
n_bin = 77
x_dim = [n_con, n_bin]
print(x_dim)

parser = ArgumentParser()
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-hDim', type=int, default=50)
parser.add_argument('-nTest', type=int, default=9000)
parser.add_argument('-zDim', type=int, default=5)
parser.add_argument('-rep', type=int, default=20)
parser.add_argument('-nIter', type=int, default=15001)
parser.add_argument('-batchSize', type=int, default=512)
parser.add_argument('-nSamplesZ', type=int, default=1)
parser.add_argument('-evalIter', type=int, default=500)
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-comment', type=str, default='')
parser.add_argument('-n_datasets', type=str, default=5)
args = parser.parse_args()

#load data
df = pd.read_csv(cwd + "/../out1_dum.csv", index_col=0)
print(df['LEVEL3'].value_counts())

#rearrange columns
# print(df.columns.tolist())
col = df.columns.tolist()
print(len(col))
col = [col[0]] + col[4:10] + [col[1]] + col[2:4] + col[42:] + col[10:14] + col[14:24] + col[24:32] + col[32:42]
# print(col)
print(len(col))
df = df[col]
pd.set_option('max_columns', 999)
# print(df.head())
print(df.columns.get_loc("LEVEL3"))
print(df.columns.get_loc("SECMPA_32.0"))
print(df.columns.get_loc("ADV_TEACH_11.0"))
print(df.columns.get_loc("ADV_TEST_11.0"))

#split train test set
df = pd.DataFrame.to_numpy(df)
#a: 0-7
#y: 7
#x_SES_con: 8-10
#x_SES_bin: 10-48
#x_lang: 48-52
#x_teach: 52-62
#x_test: 62-70
#x_fin: 70-80

a, y, x_SES_con, x_SES_bin, x_lang, x_teach, x_test, x_fin = df[:,0:7], df[:,7][:, np.newaxis], df[:,8:10], df[:,10:48], df[:,48:52], df[:,52:62], df[:,62:70], df[:,70:80]

a = a.astype(int)
y = y.astype(int)
x_SES_con = x_SES_con.astype(float)
x_SES_bin = x_SES_bin.astype(int)
x_lang = x_lang.astype(int)
x_teach = x_teach.astype(int)
x_test = x_test.astype(int)
x_fin = x_fin.astype(int)

print("Dimensions of our variables:")
print(a.shape)
print(y.shape)
print(x_SES_con.shape)
print(x_SES_bin.shape)
print(x_lang.shape)
print(x_teach.shape)
print(x_test.shape)
print(x_fin.shape)
n_obs = a.shape[0]

# data = [a, y, x_SES_con, x_SES_bin, x_lang, x_teach, x_test, x_fin]

# make an array from 0 to n observations
arr_i = np.arange(n_obs)
# use split on array as indexes for our variables
i_tr, i_te = train_test_split(arr_i, test_size= 0.1, random_state= 42)

train_data = [a[i_tr], y[i_tr], x_SES_con[i_tr], x_SES_bin[i_tr], x_lang[i_tr], x_teach[i_tr], x_test[i_tr], x_fin[i_tr]]
test_data = [a[i_te], y[i_te], x_SES_con[i_te], x_SES_bin[i_te], x_lang[i_te], x_teach[i_te], x_test[i_te], x_fin[i_te]]

print(i_tr)
print(i_te)
print(train_data[0].shape)
print(test_data[0].shape)
print(train_data[0].shape[0]/n_obs)
print(test_data[0].shape[0]/n_obs)

a_tr, y_tr, x_SES_con_tr, x_SES_bin_tr, x_lang_tr, x_teach_tr, x_test_tr, x_fin_tr = train_data
x_SES_tr = np.hstack((x_SES_con_tr, x_SES_bin_tr))

# check if cuda is available -> false
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if cuda => run on gpu

CEVAE = CEVAE_model(args=args, dim_a=a_tr.shape[1], dim_y=y_tr.shape[1], dim_s_con=x_SES_con_tr.shape[1], dim_s_bin=x_SES_bin_tr.shape[1], dim_l=x_lang_tr.shape[1], dim_t=x_teach_tr.shape[1], dim_e=x_test_tr.shape[1], dim_f=x_fin_tr.shape[1], dim_z=args.zDim, dim_q_h=args.hDim, dim_p_h=args.hDim).to(args.device)

print(args.hDim)

#init optimizer
#init loss function

#train loop:
#random batch
#forward pass network
#optimizer
#loss function

#save models
#plot loss
