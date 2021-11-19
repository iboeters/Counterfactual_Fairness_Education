import pandas as pd
import numpy as np
import os
import sys
import torch
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from CEVAE import CEVAE_model
from plot_results import plot_conf_matrix, plot_diff

# Model class for auxiliary model
class aux_m(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        # init network
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2 = nn.Linear(dim_h, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = F.elu(self.lin1(x))
        h2 = self.lin2(h1)
        out = self.softmax(h2)
        return out

def counterfactual_fairness(args, train_data, test_data):
    acc_list = defaultdict(list)
    conf_correct_list = defaultdict(list)
    conf_incorrect_list = defaultdict(list)
    count_diff = defaultdict(list)
    conf_matrix = defaultdict(list)

    np.set_printoptions(suppress=True)

    for rep_i in range(args.rep):
        print('rep: %i' % rep_i)
        a_train, y_train, s_con_train, s_bin_train, l_train, t_train, e_train, f_train = train_data

        CEVAE = CEVAE_model(args=args, dim_a=a_train.shape[1], dim_y=y_train.shape[1], dim_s_con=s_con_train.shape[1], dim_s_bin=s_bin_train.shape[1], dim_l=l_train.shape[1], dim_t=t_train.shape[1], dim_e=e_train.shape[1], dim_f=f_train.shape[1], dim_u=args.uDim, dim_q_h=args.hDim, dim_p_h=args.hDim).to(args.device)

        # load trained CEVAE model
        CEVAE.load_state_dict(torch.load('./models/CEVAE_' + args.model_name + '.pt'))

        # init Causal Path Enabler (auxiliary -fair- models)
        path_combinations = ['u', 'uf', 'usltef', 'uasltef']
        AUX = dict()
        AUX['u'] = aux_m(dim_in=args.uDim, dim_h=args.hDim, dim_out=y_train.shape[1])
        AUX['uf'] = aux_m(dim_in=args.uDim + f_train.shape[1], dim_h=args.hDim, dim_out=y_train.shape[1])
        AUX['usltef'] = aux_m(dim_in=args.uDim + s_con_train.shape[1] + s_bin_train.shape[1] + l_train.shape[1] + t_train.shape[1] + e_train.shape[1] + f_train.shape[1], dim_h=args.hDim, dim_out=y_train.shape[1])
        AUX['uasltef'] = aux_m(dim_in=args.uDim + a_train.shape[1] + s_con_train.shape[1] + s_bin_train.shape[1] + l_train.shape[1] + t_train.shape[1] + e_train.shape[1] + f_train.shape[1], dim_h=args.hDim, dim_out=y_train.shape[1])

        # init optimizers
        optimizer = dict()
        for combination in path_combinations:
            optimizer[combination] = torch.optim.RMSprop(AUX[combination].parameters(), lr=args.lr)

        # Maintain loss development for monitoring
        train_loss_dict = defaultdict(list)

        # training loop
        for i in range(args.nIter):
            # select random batch
            batch_index = np.random.choice(a=range(a_train.shape[0]), size=args.batchSize, replace=False)
            batch_data = [torch.Tensor(i[batch_index]).to(args.device) for i in train_data]
            a_batch, y_batch, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch = batch_data

            # INFER distribution over u, using inference network of CEVAE
            u_infer = CEVAE.q_u.forward(x=torch.cat((l_batch, t_batch, e_batch, f_batch), 1))
            u_infer_sample = u_infer.sample().detach()

            # forward pass networks
            aux_output = dict()
            aux_output['u'] = AUX['u'].forward(u_infer_sample)
            aux_output['uf'] = AUX['uf'].forward(torch.cat((u_infer_sample, f_batch), 1))
            aux_output['usltef'] = AUX['usltef'].forward(torch.cat((u_infer_sample, l_batch, s_batch_con, s_batch_bin, t_batch, e_batch, f_batch), 1))
            aux_output['uasltef'] = AUX['uasltef'].forward(torch.cat((u_infer_sample, a_batch, l_batch, s_batch_con, s_batch_bin, t_batch, e_batch, f_batch), 1))

            # calculate loss and update step
            loss = dict()
            for combination in path_combinations:
                loss[combination] = torch.mean((aux_output[combination] - y_batch) ** 2)
                optimizer[combination].zero_grad()
                loss[combination].backward()
                optimizer[combination].step()

                train_loss_dict[combination].append(float(loss[combination].detach().cpu().numpy()))

        # test model on test set------------------------------------------------------------------------------------
        batch_data = [torch.Tensor(i).to(args.device) for i in test_data]
        a_batch, y_batch, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch = batch_data

        # INFER distribution over u, using inference network of CEVAE
        u_infer = CEVAE.q_u.forward(x=torch.cat((l_batch, t_batch, e_batch, f_batch), 1))
        u_infer_sample = u_infer.sample().detach()
               
        aux_output = dict()
        aux_output['u'] = AUX['u'].forward(u_infer_sample)
        aux_output['uf'] = AUX['uf'].forward(torch.cat((u_infer_sample, f_batch), 1))
        aux_output['usltef'] = AUX['usltef'].forward(torch.cat((u_infer_sample, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch), 1))
        aux_output['uasltef'] = AUX['uasltef'].forward(torch.cat((u_infer_sample, a_batch, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch), 1))

        # masks
        mask_a0 = (a_batch[:,0] == 1)
        mask_a1 = (a_batch[:,1] == 1)
        mask_a2 = (a_batch[:,2] == 1)
        mask_a3 = (a_batch[:,3] == 1)

        # Save accuracy
        for combination in path_combinations:
            # calculate accuracy of algorithm, select top 2
            y_batch_max = torch.argmax(y_batch, dim=1, keepdim=True)
            values, indices = torch.topk(aux_output[combination], 2)
            correct = (indices == y_batch_max)
            accuracy = torch.sum(correct).numpy() / y_batch_max.shape[0] * 100
            acc_list[combination].append(accuracy)
            
            # calculate confidence of the model
            confidence = torch.sum(values, dim=1, keepdim=True).detach().cpu()
            one_correct = torch.sum(correct, dim=1, keepdim=True)
            none_correct = (one_correct == False)
            conf_correct= torch.sum(one_correct * confidence).detach().cpu().numpy() / torch.sum(one_correct).detach().cpu().numpy() * 100
            conf_incorrect= torch.sum(none_correct * confidence).detach().cpu().numpy() / torch.sum(none_correct).detach().cpu().numpy() * 100
            conf_correct_list[combination].append(conf_correct)
            conf_incorrect_list[combination].append(conf_incorrect)
            
            # calculate confusion matrix
            pred_y = torch.argmax(aux_output[combination], dim=1, keepdim=True).detach().cpu().numpy()
            true_y = y_batch_max.detach().cpu().numpy()
            conf = sklearn.metrics.confusion_matrix(true_y, pred_y)
            conf = conf / conf.astype(np.float).sum(axis=1)
            df_conf = pd.DataFrame(conf)
            plot_conf_matrix(df_conf, combination)

            # print equality of opportunity
            predicted = torch.argmax(aux_output[combination], dim=1, keepdim=True)
            EOP_a0 = torch.sum(predicted[mask_a0] < y_batch_max[mask_a0]).detach().cpu().numpy() / y_batch_max[mask_a0].shape[0] * 100
            EOP_a1 = torch.sum(predicted[mask_a1] < y_batch_max[mask_a1]).detach().cpu().numpy() / y_batch_max[mask_a1].shape[0] * 100
            EOP_a2 = torch.sum(predicted[mask_a2] < y_batch_max[mask_a2]).detach().cpu().numpy() / y_batch_max[mask_a2].shape[0] * 100
            EOP_a3 = torch.sum(predicted[mask_a3] < y_batch_max[mask_a3]).detach().cpu().numpy() / y_batch_max[mask_a3].shape[0] * 100
            print(combination, " EOP:\t", EOP_a0, "\t", EOP_a1, "\t", EOP_a2, "\t", EOP_a3)
            
            # print accuracy different ethnicity groups
            accuracy_a0 = torch.sum(predicted[mask_a0] == y_batch_max[mask_a0]).detach().cpu().numpy() / y_batch_max[mask_a0].shape[0] * 100
            print("accuracy a0:\t", accuracy_a0)
            accuracy_a1 = torch.sum(predicted[mask_a1] == y_batch_max[mask_a1]).detach().cpu().numpy() / y_batch_max[mask_a1].shape[0] * 100
            print("accuracy a1:\t", accuracy_a1)
            accuracy_a2 = torch.sum(predicted[mask_a2] == y_batch_max[mask_a2]).detach().cpu().numpy() / y_batch_max[mask_a2].shape[0] * 100
            print("accuracy a2:\t", accuracy_a2)
            accuracy_a3 = torch.sum(predicted[mask_a3] == y_batch_max[mask_a3]).detach().cpu().numpy() / y_batch_max[mask_a3].shape[0] * 100
            print("accuracy a3:\t", accuracy_a3)

        # Test counterfactual fairness CEVAE ------------------------------------       
        # a0 -> a1
        l_out = CEVAE.p_l_au.forward(u_infer_sample[mask_a0], a_batch[mask_a0])
        l_sample = l_out.sample().detach()
        s_bin_out, s_con_out = CEVAE.p_s_a.forward(a_batch[mask_a0], a_batch[mask_a0])
        s_bin_sample = s_bin_out.sample().detach()
        s_con_sample = s_con_out.sample().detach()
        t_out = CEVAE.p_t_alsu.forward(torch.cat((u_infer_sample[mask_a0], l_sample, s_bin_sample, s_con_sample), 1), a_batch[mask_a0])
        t_sample = t_out.sample().detach()
        e_out = CEVAE.p_e_alstu.forward(torch.cat((u_infer_sample[mask_a0], l_sample, t_sample, s_bin_sample, s_con_sample), 1), a_batch[mask_a0])
        e_sample = e_out.sample().detach()
        f_out = CEVAE.p_f_ateu.forward(torch.cat((u_infer_sample[mask_a0], t_sample, e_sample), 1), a_batch[mask_a0])
        f_sample = f_out.sample().detach()
        y_hat = CEVAE.p_y_afu.forward(torch.cat((u_infer_sample[mask_a0], f_sample), 1), a_batch[mask_a0])
        y_hat_a0 = torch.argmax(y_hat.mean, dim=1, keepdim=True)

        # a0 -> a1
        a_batch_a_ = a_batch[mask_a0]
        a_batch_a_[:,0] = 0
        a_batch_a_[:,1] = 1
        l_out = CEVAE.p_l_au.forward(u_infer_sample[mask_a0], a_batch_a_)
        l_sample = l_out.sample().detach()
        s_bin_out, s_con_out = CEVAE.p_s_a.forward(a_batch_a_, a_batch_a_)
        s_bin_sample = s_bin_out.sample().detach()
        s_con_sample = s_con_out.sample().detach()
        t_out = CEVAE.p_t_alsu.forward(torch.cat((u_infer_sample[mask_a0], l_sample, s_bin_sample, s_con_sample), 1), a_batch_a_)
        t_sample = t_out.sample().detach()
        e_out = CEVAE.p_e_alstu.forward(torch.cat((u_infer_sample[mask_a0], l_sample, t_sample, s_bin_sample, s_con_sample), 1), a_batch_a_)
        e_sample = e_out.sample().detach()
        f_out = CEVAE.p_f_ateu.forward(torch.cat((u_infer_sample[mask_a0], t_sample, e_sample), 1), a_batch_a_)
        f_sample = f_out.sample().detach()
        y_hat = CEVAE.p_y_afu.forward(torch.cat((u_infer_sample[mask_a0], f_sample), 1), a_batch_a_)
        y_hat_a1 = torch.argmax(y_hat.mean, dim=1, keepdim=True)   
  
        # calculate difference counterfactual
        diff = y_hat_a0 - y_hat_a1
        diff_pos = torch.pow(diff, 2)
        diff_pos = torch.sqrt(diff_pos.float())
        diff_mean = torch.mean(diff_pos.float()).detach().cpu().numpy()

        diff_array = diff.detach().cpu().numpy()
        count_diff[rep_i].append(diff_array)

    df = pd.DataFrame(diff_array, index=np.arange(diff_array.shape[0]), columns=["diff"])
    plot_diff(df, args)    
    
    print("\nAccuracies")
    for (key, value), (k2, v2), (k3, v3) in zip(acc_list.items(), conf_correct_list.items(), conf_incorrect_list.items()):
        print(key, "\t", value)
        print("\t", v2)
        print("\t", v3)
        print("acc:\t", np.mean(value), "\tsd:", np.std(value))
        print("conf_c:\t", np.mean(v2), "\tsd:", np.std(v2))
        print("conf_i:\t", np.mean(v3), "\tsd:", np.std(v3))

    np.save('./output/accuracy_dict_' + args.filename + '.npy', acc_list)
