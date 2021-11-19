import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

def get_data(dataset):
    # get current working directory
    cwd = os.getcwd()

    # load data
    df = pd.read_csv(cwd + "/../output/" + dataset, index_col=0)
    
    # rearrange columns to: gender->ethn->level3->SES->language->advices
    col = df.columns.tolist()
    col = [col[0]] + col[3:10] + col[10:16] + col[1:3] + col[48:] + col[16:20] + col[20:30] + col[30:38] + col[38:48]
    df = df[col]

    # a = sensitive attributes, y=level after 3 yrs, s_con=SES continuous vars, s_bin=SES binary variables, l=language level, t=teachers advice, e=EPST test advice, f=final advice
    df = pd.DataFrame.to_numpy(df)
    a, y, s_con, s_bin, l, t, e, f = df[:,0:8], df[:,8:14], df[:,14:16], df[:,16:54], df[:,54:58], df[:,58:68], df[:,68:76], df[:,76:86]

    # uncomment for make binary a_var
    # a = a[:,1][:,np.newaxis]
    # make 4 categories a
    a[:,0] = a[:,1]
    a[:,1] = a[:,2] + a[:,3] + a[:,4]
    a[:,2] = a[:,5]
    a[:,3] = a[:,6] + a[:,7]
    a = a[:, 0:4]
    
    a = a.astype(int)
    y = y.astype(int)
    s_con = s_con.astype(float)
    s_bin = s_bin.astype(int)
    l = l.astype(int)
    t = t.astype(int)
    e = e.astype(int)
    f = f.astype(int)
    
    # uncomment for printing Dimensions of variables
    # print("\nDimensions of our variables:")
    # print(a.shape)
    # print(y.shape)
    # print(s_con.shape)
    # print(s_bin.shape)
    # print(l.shape)
    # print(t.shape)
    # print(e.shape)
    # print(f.shape)

    data = [a, y, s_con, s_bin, l, t, e, f]
    return data

def get_train_test_split(data, test_size):
    a, y, s_con, s_bin, l, t, e, f = data
    n_obs = a.shape[0]

    # make an array from 0 to n observations
    arr_i = np.arange(n_obs)
    # use split on array as indexes for our variables
    i_train, i_test = train_test_split(arr_i, test_size=test_size, random_state= 42)

    train_data = [a[i_train], y[i_train], s_con[i_train], s_bin[i_train], l[i_train], t[i_train], e[i_train], f[i_train]]
    test_data = [a[i_test], y[i_test], s_con[i_test], s_bin[i_test], l[i_test], t[i_test], e[i_test], f[i_test]]

    return train_data, test_data
