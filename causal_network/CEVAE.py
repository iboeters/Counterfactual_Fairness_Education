import torch
import torch.nn as nn
# holds convolution functions; all functions without parameters; relu
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# different classes:
# q(u|l, t, e, f)
# p(u)
# p(l|a, u)
# p(s|a)
# p(t|a, l, s, u)
# p(e|a, l, s, t, u)
# p(f|a, t, e, u)
# p(y|a, f, u)

# inherit modules from nn.Module
class q_u_ltef(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_h)
        self.mu = nn.Linear(dim_h, dim_out)
        self.sigma = nn.Linear(dim_h, dim_out)
        # softplus function for positive output
        self.softplus = nn.Softplus()

    def forward(self, x):
        h1 = F.elu(self.lin(x))
        mu = self.mu(h1)
        # variance is forced to have a positive output
        sigma = self.softplus(self.sigma(h1))
        out = normal.Normal(mu, sigma)

        return out

# p(l|a, u)
class p_l_au_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.lin2_a2 = nn.Linear(dim_h, dim_out)
        self.lin2_a3 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)
        
        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = torch.sigmoid(h2_a2)
        bern_p_a3 = torch.sigmoid(h2_a3)

        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)

        return bern_out
    
# p(s|a)
class p_s_a_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out_bin, dim_out_con):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out_bin)
        self.lin2_a1 = nn.Linear(dim_h, dim_out_bin)
        self.lin2_a2 = nn.Linear(dim_h, dim_out_bin)
        self.lin2_a3 = nn.Linear(dim_h, dim_out_bin)

        self.mu_a0 = nn.Linear(dim_h, dim_out_con)
        self.mu_a1 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a0 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a1 = nn.Linear(dim_h, dim_out_con)
        self.mu_a2 = nn.Linear(dim_h, dim_out_con)
        self.mu_a3 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a2 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a3 = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = torch.sigmoid(h2_a2)
        bern_p_a3 = torch.sigmoid(h2_a3)
        
        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)
        
        mu_a0 = self.mu_a0(h1)
        mu_a1 = self.mu_a1(h1)
        sigma_a0 = self.softplus(self.sigma_a0(h1))
        sigma_a1 = self.softplus(self.sigma_a1(h1))
        mu_a2 = self.mu_a2(h1)
        mu_a3 = self.mu_a3(h1)
        sigma_a2 = self.softplus(self.sigma_a2(h1))
        sigma_a3 = self.softplus(self.sigma_a3(h1))

        con_out = normal.Normal(a_0 * mu_a0 + a_1 * mu_a1 + a_2 * mu_a2 + a_3 * mu_a3, a_0 * sigma_a0 + a_1 * sigma_a1 + a_2 * sigma_a2 + a_3 * sigma_a3)

        return bern_out, con_out

# p(t|s, a, l, u)
class p_t_alsu_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.lin2_a2 = nn.Linear(dim_h, dim_out)
        self.lin2_a3 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = torch.sigmoid(h2_a2)
        bern_p_a3 = torch.sigmoid(h2_a3)
        
        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)
        return bern_out

# p(e|s, a, l, t, u)
class p_e_alstu_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.lin2_a2 = nn.Linear(dim_h, dim_out)
        self.lin2_a3 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = torch.sigmoid(h2_a2)
        bern_p_a3 = torch.sigmoid(h2_a3)
        
        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)

        return bern_out

# p(f|a, t, e, u)
class p_f_ateu_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.lin2_a2 = nn.Linear(dim_h, dim_out)
        self.lin2_a3 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = torch.sigmoid(h2_a2)
        bern_p_a3 = torch.sigmoid(h2_a3)
        
        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)

        return bern_out

# p(y|a, f, u)
class p_y_afu_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.lin2_a2 = nn.Linear(dim_h, dim_out)
        self.lin2_a3 = nn.Linear(dim_h, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = self.softmax(h2_a0)
        bern_p_a1 = self.softmax(h2_a1)

        h2_a2 = F.elu(self.lin2_a2(h1))
        h2_a3 = F.elu(self.lin2_a3(h1))
        bern_p_a2 = self.softmax(h2_a2)
        bern_p_a3 = self.softmax(h2_a3)

        a_0 = (a[:,0] == 1)[:,np.newaxis]
        a_1 = (a[:,1] == 1)[:,np.newaxis]
        a_2 = (a[:,2] == 1)[:,np.newaxis]
        a_3 = (a[:,3] == 1)[:,np.newaxis]
        bern_out = bernoulli.Bernoulli(a_0 * bern_p_a0 + a_1 * bern_p_a1 + a_2 * bern_p_a2 + a_3 * bern_p_a3)

        return bern_out


class CEVAE_model(nn.Module):
    def __init__(self, args, dim_a, dim_y, dim_s_con, dim_s_bin, dim_l, dim_t, dim_e, dim_f, dim_u, dim_q_h=100, dim_p_h=50):
        super().__init__()
        self.q_u = q_u_ltef(dim_in=dim_l + dim_t + dim_e + dim_f, dim_h=dim_q_h, dim_out=dim_u)
        self.p_l_au = p_l_au_model(dim_in=dim_u, dim_h=dim_p_h, dim_out=dim_l)
        self.p_s_a = p_s_a_model(dim_in=dim_a, dim_h=dim_p_h, dim_out_bin=dim_s_bin, dim_out_con=dim_s_con)
        self.p_t_alsu = p_t_alsu_model(dim_in=dim_s_con + dim_s_bin + dim_l + dim_u, dim_h=dim_p_h, dim_out=dim_t)
        self.p_e_alstu = p_e_alstu_model(dim_in=dim_l+dim_t+dim_s_con+dim_s_bin+dim_u, dim_h=dim_p_h, dim_out=dim_e)
        self.p_f_ateu = p_f_ateu_model(dim_in=dim_t+dim_e+dim_u, dim_h=dim_p_h, dim_out=dim_t)
        self.p_y_afu = p_y_afu_model(dim_in=dim_f + dim_u, dim_h=dim_p_h, dim_out=dim_y)
        self.p_u = normal.Normal(torch.zeros(args.uDim).to(args.device), torch.ones(args.uDim).to(args.device))

    def forward(self, args, batch_data, train_loss_dict, eval=False, reconstruct=False, switch_a=False):
        # get batch data
        a_batch, y_batch, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch = batch_data

        # q(u|l, t, e, f)
        u_infer = self.q_u.forward(x=torch.cat((l_batch, t_batch, e_batch, f_batch), 1))

        if eval:
            u_infer_sample = u_infer.mean
        else:
            u_infer_sample = u_infer.rsample() # for reparemetrization trick, allows for pathwise derivatives

        # reconstruction loss (first term loss function)
        # take average log probability: information entropy
        l_out = self.p_l_au.forward(u_infer_sample, a_batch)
        loss_l = l_out.log_prob(l_batch).sum(axis=1)
        train_loss_dict['Reconstruction_l'].append(loss_l.mean().cpu().detach().float())

        s_bin_out, s_con_out = self.p_s_a.forward(a_batch, a_batch)
        loss_s1 = s_con_out.log_prob(s_batch_con).sum(axis=1)
        train_loss_dict['Reconstruction_s_con'].append(loss_s1.mean().cpu().detach().float())
        loss_s2 = s_bin_out.log_prob(s_batch_bin).sum(axis=1)
        train_loss_dict['Reconstruction_s_bin'].append(loss_s2.mean().cpu().detach().float())
        
        t_out = self.p_t_alsu.forward(torch.cat((u_infer_sample, l_batch, s_batch_bin, s_batch_con), 1), a_batch)
        loss_t = t_out.log_prob(t_batch).sum(axis=1)
        train_loss_dict['Reconstruction_t'].append(loss_t.mean().cpu().detach().float())
        
        e_out = self.p_e_alstu.forward(torch.cat((u_infer_sample, l_batch, t_batch, s_batch_bin, s_batch_con), 1), a_batch)
        loss_e = e_out.log_prob(e_batch).sum(axis=1)
        train_loss_dict['Reconstruction_e'].append(loss_e.mean().cpu().detach().float())
        
        f_out = self.p_f_ateu.forward(torch.cat((u_infer_sample, t_batch, e_batch), 1), a_batch)
        loss_f = f_out.log_prob(f_batch).sum(axis=1)
        train_loss_dict['Reconstruction_f'].append(loss_f.mean().cpu().detach().float())
        
        y_infer = self.p_y_afu.forward(torch.cat((u_infer_sample, f_batch), 1), a_batch)
        loss_y = y_infer.log_prob(y_batch).sum(axis=1)
        train_loss_dict['Reconstruction_y'].append(loss_y.mean().cpu().detach().float())
        
        # regularization loss (second term loss function)
        # Kullback-Leibler divergence between encoders distribution q phi (u|x) and p(u)- normal distribution
        loss_reg = (self.p_u.log_prob(u_infer_sample) - u_infer.log_prob(u_infer_sample)).sum(axis=1)
        train_loss_dict['Regularization'].append(loss_reg.mean().detach().cpu().float())

        # total loss VAE: negative log likelihood + regularizer
        loss_rec = loss_l + loss_s1 + loss_s2 + loss_t + loss_e + loss_f + loss_y
        loss_total = -torch.mean(loss_rec + loss_reg)
        train_loss_dict['Total loss'].append(loss_total.cpu().detach().numpy())

        if eval:
            y_infer_max = torch.argmax(y_infer.mean, dim=1, keepdim=True)
            y_batch_max = torch.argmax(y_batch, dim=1, keepdim=True)
            accuracy = torch.sum(y_infer_max == y_batch_max).cpu().detach().numpy() / y_batch_max.shape[0] * 100
            return accuracy, loss_total, u_infer_sample

        return loss_total, train_loss_dict, loss_reg, loss_rec
