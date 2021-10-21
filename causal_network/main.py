import pandas as pd
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from collections import defaultdict
from argparse import ArgumentParser
from CEVAE import CEVAE_model, scatter_latent
from fairness_unawareness import fairness_unawareness

# to run: python main.py -filename=bla -nIter=50 -evalIter=10
# python main.py -filename=bla -nIter=1000 -evalIter=100


pd.set_option('max_columns', 999)
cwd = os.getcwd()

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
parser.add_argument('-filename', type=str, default='test')
args = parser.parse_args()

# load data
df = pd.read_csv(cwd + "/../output/out1_dum.csv", index_col=0)
# print(df['LEVEL3'].value_counts())

# rearrange columns
# print(df.columns.tolist())
print("Locate columns:")
print(df.columns.get_loc("LEVEL3_11.0"))
print(df.columns.get_loc("SECMPA_32.0"))
print(df.columns.get_loc("WOZ"))
print(df.columns.get_loc("ETHN_6.0"))
print(df.columns.get_loc("WPOTAALTV_2.0"))
print(df.columns.get_loc("ADV_TEACH_11.0"))
print(df.columns.get_loc("ADV_TEST_11.0"))
print(df.columns.get_loc("ADV_FINAL_11.0"))
col = df.columns.tolist()
# print(col)
print(len(col))
#gender;ethn;level3;SES;language;advices
col = [col[0]] + col[3:10] + col[10:16] + col[1:3] + col[48:] + col[16:20] + col[20:30] + col[30:38] + col[38:48]
# print(col)
print(len(col))
df = df[col]

# split train test set
# a = sensitive attributes, y=level after 3 yrs, s_con=SES continuous vars, s_bin=SES binary variables, l=language level, t=teachers advice, e=EPST test advice, f=final advice
# a: 0-7
# y: 8-13
# s_con: 14-15
# s_bin: 16-53
# l: 54-57
# t: 58-67
# e: 68-75
# f: 76-87
df = pd.DataFrame.to_numpy(df)
a, y, s_con, s_bin, l, t, e, f = df[:,0:8], df[:,8:14], df[:,14:16], df[:,16:54], df[:,54:58], df[:,58:68], df[:,68:76], df[:,76:86]

# make binary a_var
a = a[:,1][:,np.newaxis]
print("\nMean a:")
print(np.mean(a))

# print outcome variable
print("y outcome")
print(np.sum(y, axis=0))
print("percentages:")
print(np.sum(y, axis=0) / y.shape[0] * 100)

a = a.astype(int)
y = y.astype(int)
s_con = s_con.astype(float)
s_bin = s_bin.astype(int)
l = l.astype(int)
t = t.astype(int)
e = e.astype(int)
f = f.astype(int)

print("\nDimensions of our variables:")
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

print("\nPrint split info")
print(i_train)
print(i_test)
print(train_data[0].shape)
print(test_data[0].shape)
print(train_data[0].shape[0]/n_obs)
print(test_data[0].shape[0]/n_obs)

a_train, y_train, s_con_train, s_bin_train, l_train, t_train, e_train, f_train = train_data
a_test, y_test, s_con_test, s_bin_test, l_test, t_test, e_test, f_test = test_data
x_SES_train = np.hstack((s_con_train, s_bin_train))
# y_train = F.one_hot(y_train, num_classes = 5)
# y_test = F.one_hot(y_test, num_classes = 5)
# print("one hot encoding results:")
# print(y_train)
# print(y_test)

# check if cuda is available; if cuda => run on gpu
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 



# fairness through unawareness --------------------------
# reverse dummy vars y
# y = pd.DataFrame(data = y, columns = ["1", "3", "5", "7", "9", "11"])
# y = y.idxmax(axis=1)
# y = pd.DataFrame.to_numpy(y)
# y = y[:][:,np.newaxis]

# print(n_obs)
# arr_i = np.arange(n_obs)
# # use split on array as indexes for our variables
# i_train, i_test = train_test_split(arr_i, test_size= 0.1, random_state= 42)
# train_data = [a[i_train], y[i_train], s_con[i_train], s_bin[i_train], l[i_train], t[i_train], e[i_train], f[i_train]]
# test_data = [a[i_test], y[i_test], s_con[i_test], s_bin[i_test], l[i_test], t[i_test], e[i_test], f[i_test]]

# a_train, y_train, s_con_train, s_bin_train, l_train, t_train, e_train, f_train = train_data
# a_test, y_test, s_con_test, s_bin_test, l_test, t_test, e_test, f_test = test_data

# x_train = np.concatenate((s_con_train, s_bin_train, l_train, t_train, e_train, f_train), axis=1)
# x_test = np.concatenate((s_con_test, s_bin_test, l_test, t_test, e_test, f_test), axis=1)
# fairness_unawareness(args, x_train, x_test, y_train, y_test)
# exit(1)



# CEVAE model -------------------------------------------
CEVAE = CEVAE_model(args=args, dim_a=a_train.shape[1], dim_y=y_train.shape[1], dim_s_con=s_con_train.shape[1], dim_s_bin=s_bin_train.shape[1], dim_l=l_train.shape[1], dim_t=t_train.shape[1], dim_e=e_train.shape[1], dim_f=f_train.shape[1], dim_z=args.zDim, dim_q_h=args.hDim, dim_p_h=args.hDim).to(args.device)


# init optimizer
optimizer = torch.optim.Adam(CEVAE.parameters(), lr=args.lr)
# variables to save loss result
loss_dict = defaultdict(list)
test_loss_dict = defaultdict(list)

# train loop
for i in range(args.nIter):
    # random batch
    if args.batchSize > a_train.shape[0]:
        print("Error: batchSize [%f] bigger than the training dataset [%f]" % (args.batchSize, a_train.shape[0]))
        exit(1)
    batch_index = np.random.choice(a=range(a_train.shape[0]), size=args.batchSize, replace=False)
    batch_data = [torch.Tensor(i[batch_index]).to(args.device) for i in train_data]

    # forward pass network
    loss, loss_dict = CEVAE.forward(args, batch_data, loss_dict)
    
    # clear gradients optimizer
    optimizer.zero_grad()
    # 
    loss.backward()
    # optimizer
    optimizer.step()

    # save loss training
    loss_dict['train_loss'].append(loss.cpu().detach().numpy())

    # show progress every evalIter times
    if i % args.evalIter == 0:
        print('\nIteration: %i / %i ' % (i, args.nIter))
        # test whole test set
        batch_data = [torch.Tensor(j).to(args.device) for j in test_data]
        accuracy, test_loss, z_mean = CEVAE.forward(args, batch_data, test_loss_dict, eval=True)
        test_loss_dict['test_loss'].append(test_loss.cpu().detach().numpy())
        test_loss_dict['test_loss_index'].append(i)
        print('Accuracy test set: %f' % accuracy)
        print('Test  loss: %f' % test_loss)
        print('Train loss: %f' % loss.cpu().detach().numpy())

        # scatter latent for independence indication
#         if i == 0 or i % 500 == 0:
#             a_batch = batch_data[0]
#             scatter_latent(z=z_mean, condition=a_batch, i=i)

        # save model; outside of loop?
        torch.save(CEVAE.state_dict(), 'models/CEVAE_' + args.filename + '.pt')



# Plot results ----------------------------------------
np.save('output/train_loss_develop_' + args.filename + '.npy', loss_dict['train_loss'])
np.save('output/test_loss_develop_' + args.filename + '.npy', test_loss_dict['test_loss'])
np.save('output/y_train_loss_' + args.filename + '.npy', loss_dict['Reconstructing_y'])
np.save('output/y_test_loss_' + args.filename + '.npy', test_loss_dict['Reconstructing_y'])

plt.figure(figsize=(18.0, 12.0))
plt.plot(loss_dict['train_loss'], label='train')
plt.plot(test_loss_dict['test_loss_index'], test_loss_dict['test_loss'], label='test')
plt.legend()
plt.title('Loss development training CEVAE')
plt.tight_layout()
plt.savefig('output/loss_develop_' + args.filename + '.png')
plt.close()
