import pandas as pd
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from CEVAE import CEVAE_model
from plot_results import plot_loss

def train_CEVAE(args, train_data, test_data):
    # get data
    a_train, y_train, s_con_train, s_bin_train, l_train, t_train, e_train, f_train = train_data
    a_test, y_test, s_con_test, s_bin_test, l_test, t_test, e_test, f_test = test_data

    # init CEVAE_model
    CEVAE = CEVAE_model(args=args, dim_a=a_train.shape[1], dim_y=y_train.shape[1], dim_s_con=s_con_train.shape[1], dim_s_bin=s_bin_train.shape[1], dim_l=l_train.shape[1], dim_t=t_train.shape[1], dim_e=e_train.shape[1], dim_f=f_train.shape[1], dim_u=args.uDim, dim_q_h=args.hDim, dim_p_h=args.hDim).to(args.device)

    # init optimizer
    optimizer = torch.optim.Adam(CEVAE.parameters(), lr=args.lr)

    # variables to save loss result
    train_loss_dict = defaultdict(list)
    test_loss_dict = defaultdict(list)

    # training loop
    for i in range(args.nIter_CEVAE):
        # random batch
        if args.batchSize > a_train.shape[0]:
            print("Error: batchSize [%f] bigger than the training dataset [%f]" % (args.batchSize, a_train.shape[0]))
            exit(1)
        batch_index = np.random.choice(a=range(a_train.shape[0]), size=args.batchSize, replace=False)
        batch_data = [torch.Tensor(i[batch_index]).to(args.device) for i in train_data]

        # forward pass network
        loss, train_loss_dict, loss_reg, loss_rec = CEVAE.forward(args, batch_data, train_loss_dict)

        # clear gradients optimizer
        optimizer.zero_grad()

        loss.backward()
        # optimizer
        optimizer.step()

        # save loss training
        train_loss_dict['train_loss'].append(loss.detach().cpu().numpy())
        train_loss_dict['index'].append(i)

        # show progress every evalIter times
        if i % args.evalIter == 0 or i + 1 == args.nIter_CEVAE:
            print('\nIteration: %i / %i ' % (i, args.nIter_CEVAE))

            # test model on whole test set
            batch_data = [torch.Tensor(j).to(args.device) for j in test_data]
            accuracy, test_loss, u_mean = CEVAE.forward(args, batch_data, test_loss_dict, eval=True)
            test_loss_dict['test_loss'].append(test_loss.detach().cpu().numpy())
            test_loss_dict['index'].append(i)
            print('Accuracy test set: %f' % accuracy)
            print('Test  loss: %f' % test_loss)
            print('Train loss: %f' % loss.detach().cpu().numpy())

    # plot_loss
    plot_loss(args, train_loss_dict, test_loss_dict)

    # save losses
    np.save('output/train_loss_develop_' + args.filename + '.npy', train_loss_dict['train_loss'])
    np.save('output/test_loss_develop_' + args.filename + '.npy', test_loss_dict['test_loss'])
    np.save('output/y_train_loss_' + args.filename + '.npy', train_loss_dict['Reconstruction_y'])
    np.save('output/y_test_loss_' + args.filename + '.npy', test_loss_dict['Reconstruction_y'])

    # save weights and biases of the learned CEVAE model
    torch.save(CEVAE.state_dict(), 'models/CEVAE_' + args.filename + '.pt')
