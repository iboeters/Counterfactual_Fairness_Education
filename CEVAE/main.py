import pandas as pd
import numpy as np
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from argparse import ArgumentParser
from CEVAE import CEVAE_model
from fairness_unaware import fairness_unaware

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
#a = sensitive attributes, y=level after 3 yrs, s_con=SES continuous vars, s_bin=SES binary variables, l=language level, t=teachers advice, e=EPST test advice, f=final advice
#a: 0-7
#y: 7
#s_con: 8-10
#s_bin: 10-48
#l: 48-52
#t: 52-62
#e: 62-70
#f: 70-80

a, y, s_con, s_bin, l, t, e, f = df[:,0:7], df[:,7][:, np.newaxis], df[:,8:10], df[:,10:48], df[:,48:52], df[:,52:62], df[:,62:70], df[:,70:80]

a = a.astype(int)
y = y.astype(int)
s_con = s_con.astype(float)
s_bin = s_bin.astype(int)
l = l.astype(int)
t = t.astype(int)
e = e.astype(int)
f = f.astype(int)

print("Dimensions of our variables:")
print(a.shape)
print(y.shape)
print(s_con.shape)
print(s_bin.shape)
print(l.shape)
print(t.shape)
print(e.shape)
print(f.shape)
n_obs = a.shape[0]

# data = [a, y, s_con, s_bin, l, t, e, f]

# make an array from 0 to n observations
arr_i = np.arange(n_obs)
# use split on array as indexes for our variables
i_train, i_test = train_test_split(arr_i, test_size= 0.1, random_state= 42)

train_data = [a[i_train], y[i_train], s_con[i_train], s_bin[i_train], l[i_train], t[i_train], e[i_train], f[i_train]]
test_data = [a[i_test], y[i_test], s_con[i_test], s_bin[i_test], l[i_test], t[i_test], e[i_test], f[i_test]]

print(i_train)
print(i_test)
print(train_data[0].shape)
print(test_data[0].shape)
print(train_data[0].shape[0]/n_obs)
print(test_data[0].shape[0]/n_obs)

a_train, y_train, s_con_train, s_bin_train, l_train, t_train, e_train, f_train = train_data
a_test, y_test, s_con_test, s_bin_test, l_test, t_test, e_test, f_test = test_data
x_SES_train = np.hstack((s_con_train, s_bin_train))

# check if cuda is available -> false
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if cuda => run on gpu

x_train = s_con_train, s_bin_train, l_train, t_train, e_train, f_train
x_test = s_con_test, s_bin_test, l_test, t_test, e_test, f_test
# fairness through unawareness
fairness_unaware(args, x_train, x_test, y_train, y_test)

CEVAE = CEVAE_model(args=args, dim_a=a_train.shape[1], dim_y=y_train.shape[1], dim_s_con=s_con_train.shape[1], dim_s_bin=s_bin_train.shape[1], dim_l=l_train.shape[1], dim_t=t_train.shape[1], dim_e=e_train.shape[1], dim_f=f_train.shape[1], dim_z=args.zDim, dim_q_h=args.hDim, dim_p_h=args.hDim).to(args.device)

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


