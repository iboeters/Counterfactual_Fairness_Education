import torch
import torch.nn as nn
# holds convolution functions; all functions without parameters; relu
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# different classes:
# q(z)
# p(l|a, z)
# p(s|a)
# p(t|s, a, l, z)
# p(e|s, a, l, t, z)
# p(f|a, t, e, z) ; add later??
# p(y|a, f, z)

# h = n hidden layers
# dim_h = dimension hidden layers

# inherit modules from nn.Module
class q_z_yltef(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        # call constructor of parent -> const and variables + methods
        super().__init__()
        # 3 linear neural networks Linear(size_input, size_output)
        # apply linear transformation to incoming data
        self.lin = nn.Linear(dim_in, dim_h)
        self.mu = nn.Linear(dim_h, dim_out)
        self.sigma = nn.Linear(dim_h, dim_out)
        # softplus function for positive output
        self.softplus = nn.Softplus()

    def forward(self, x):
        # 1 hidden layer, 100 nodes = dim_h, with elu activation
        h1 = F.elu(self.lin(x))
        mu = self.mu(h1)
        # variance is forced to have a positive output
        sigma = self.softplus(self.sigma(h1))
        out = normal.Normal(mu, sigma)

        return out

# p(l|a, z)
class p_l_az_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        # bernoulli distribution for a0 and a1 seperately
        # if a = 0 -> return p_a0 if 1 -> return p_a1
        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out
    
# p(s|a)
class p_s_a_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out_bin, dim_out_con):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out_bin)
        self.lin2_a1 = nn.Linear(dim_h, dim_out_bin)

        self.mu_a0 = nn.Linear(dim_h, dim_out_con)
        self.mu_a1 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a0 = nn.Linear(dim_h, dim_out_con)
        self.sigma_a1 = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)
        
        mu_a0 = self.mu_a0(h1)
        mu_a1 = self.mu_a1(h1)
        sigma_a0 = self.softplus(self.sigma_a0(h1))
        sigma_a1 = self.softplus(self.sigma_a1(h1))
        con_out = normal.Normal((1-a) * mu_a0 + a * mu_a1, (1-a)* sigma_a0 + a * sigma_a1)

        return bern_out, con_out

# p(t|s, a, l, z)
class p_t_alsz_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out

# p(e|s, a, l, t, z)
class p_e_altsz_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out

# p(f|a, t, e, z)
class p_f_atez_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out

# p(y|a, f, z)
class p_y_afz_model(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = self.softmax(h2_a0)
        bern_p_a1 = self.softmax(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out


class CEVAE_model(nn.Module):
    def __init__(self, args, dim_a, dim_y, dim_s_con, dim_s_bin, dim_l, dim_t, dim_e, dim_f, dim_z, dim_q_h=100, dim_p_h=50):
        super().__init__()
        # Change structure in case of resolving variables
        self.q_z = q_z_yltef(dim_in=dim_y + dim_l + dim_t + dim_e + dim_f, dim_h=dim_q_h, dim_out=dim_z)
        self.p_l_az = p_l_az_model(dim_in=dim_a + dim_z, dim_h=dim_p_h, dim_out=dim_l)
        self.p_s_a = p_s_a_model(dim_in=dim_a, dim_h=dim_p_h, dim_out_bin=dim_s_bin, dim_out_con=dim_s_con)
        self.p_t_aslz = p_t_alsz_model(dim_in=dim_a + dim_s_con + dim_s_bin + dim_l + dim_z, dim_h=dim_p_h, dim_out=dim_t)
        self.p_e_altsz = p_e_altsz_model(dim_in=dim_a+dim_l+dim_t+dim_s_con+dim_s_bin+dim_z, dim_h=dim_p_h, dim_out=dim_e)
        self.p_f_atez = p_f_atez_model(dim_in=dim_a+dim_t+dim_e+dim_z, dim_h=dim_p_h, dim_out=dim_t)
        self.p_y_afz = p_y_afz_model(dim_in=dim_a + dim_f + dim_z, dim_h=dim_p_h, dim_out=dim_y)
        self.p_z_dist = normal.Normal(torch.zeros(args.zDim).to(args.device), torch.ones(args.zDim).to(args.device))

    def forward(self, args, batch_data, loss_dict, eval=False, reconstruct=False, switch_a=False):
        # save batch data
        a_batch, y_batch, s_batch_con, s_batch_bin, l_batch, t_batch, e_batch, f_batch = batch_data

        # q(z|a, b, x, r)
        z_infer = self.q_z.forward(x=torch.cat((y_batch, l_batch, t_batch, e_batch, f_batch), 1))
        
        if eval:
            z_infer_sample = z_infer.mean
        else:
            z_infer_sample = z_infer.rsample()

        if switch_a:
            a_batch = 1 - a_batch

        # reconstruction loss (first term loss function)
        l_out = self.p_l_az.forward(torch.cat((z_infer_sample, a_batch), 1), a_batch)
        loss_l = l_out.log_prob(l_batch).sum(1)
        loss_dict['Reconstruction_l'].append(loss_l.mean().cpu().detach().float())
        
        s_bin_out, s_con_out = self.p_s_a.forward(a_batch, a_batch)
        loss_s1 = s_con_out.log_prob(s_batch_con).sum(1)
        loss_dict['Reconstruction_s_con'].append(loss_s1.mean().cpu().detach().float())
        
        loss_s2 = s_bin_out.log_prob(s_batch_bin).sum(1)
        loss_dict['Reconstruction_s_bin'].append(loss_s2.mean().cpu().detach().float())
        
        t_out = self.p_t_aslz.forward(torch.cat((z_infer_sample, l_batch, s_batch_bin, s_batch_con, a_batch), 1), a_batch)
        loss_t = t_out.log_prob(t_batch).sum(1)
        loss_dict['Reconstruction_t'].append(loss_t.mean().cpu().detach().float())
        
        e_out = self.p_e_altsz.forward(torch.cat((z_infer_sample, l_batch, t_batch, s_batch_bin, s_batch_con, a_batch), 1), a_batch)
        loss_e = e_out.log_prob(e_batch).sum(1)
        loss_dict['Reconstruction_e'].append(loss_e.mean().cpu().detach().float())
        
        f_out = self.p_f_atez.forward(torch.cat((z_infer_sample, t_batch, e_batch, a_batch), 1), a_batch)
        loss_f = f_out.log_prob(f_batch).sum(1)
        loss_dict['Reconstruction_f'].append(loss_f.mean().cpu().detach().float())
        
        y_infer = self.p_y_afz.forward(torch.cat((z_infer_sample, f_batch, a_batch), 1), a_batch)
        # take average log probability: information entropy
        loss_y = y_infer.log_prob(y_batch).sum(1)
        loss_dict['Reconstruction_y'].append(loss_y.mean().cpu().detach().float())
        
        # regularization loss (second term loss function)
        # Kullback-Leibler divergence between encoders distribution q phi (z|x) and p(z)
        # -> difference between q and p
        loss_z = (self.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
        loss_dict['Regularization'].append(loss_z.mean().cpu().detach().float())

        # total loss VAE is negative log likelihood + regularizer
        loss_total = -torch.mean(loss_l + loss_s1 + loss_s2 + loss_t + loss_e + loss_f + loss_y + loss_z)
        loss_dict['Total loss'].append(loss_total.cpu().detach().numpy())
        
        if eval:
#             print("f_out:")
#             print(f_out.probs)
#             print("y_infer log prob:")
#             print(loss_y)
#             print("\ny_batch:")
#             print(y_batch)
#             print("unique counts:")
#             uni, n = torch.unique(y_batch, return_counts=True)
#             print(n)
#             print((n.numpy()[0] / torch.sum(n).item()) * 100)
#             print("\ny_infer:")
#             print(y_infer)
#             print(y_infer.probs)
#             print(y_infer.probs.size())
#             print(y_infer.logits)
#             print(y_infer.logits.size())
#             print(y_infer.mean)
#             print(y_infer.mean.size())
#             print("\n argmax:")
#             print(torch.argmax(y_infer.mean, dim=1, keepdim=True))
#             y_out = torch.round(y_infer.mean)
#             print(torch.argmax(y_batch, dim=1, keepdim=True))
#             print("y_out:")
#             print(y_out)
#             print(torch.sum(y_out == y_batch))
#             print("y_out shape:")
#             print(y_out.shape)
#             print("y_batch shape:")
#             print(y_batch.shape)
            y_infer_max = torch.argmax(y_infer.mean, dim=1, keepdim=True)
            y_batch_max = torch.argmax(y_batch, dim=1, keepdim=True)
            accuracy = torch.sum(y_infer_max == y_batch_max).cpu().detach().numpy() / y_batch_max.shape[0] * 100
            return accuracy, loss_total, z_infer_sample

        # when Reconstruction, return reconstructed distributions
        if reconstruct:
            reconstruction = {
                'l': l_out,
                's_con': s_con_out,
                's_bin': s_bin_out,
                't': t_out,
                'e': e_out,
                'f': f_out,
                'y': y_infer
            }
            return reconstruction

        return loss_total, loss_dict

def scatter_latent(z, condition, i):
    # plot scatter z, for insight independence Z, A
    z_tsne = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
    plt.figure(figsize=(4, 4))
    plt.plot(z_tsne[np.where(condition.cpu().detach().numpy() == 0)[0], 0],
                z_tsne[np.where(condition.cpu().detach().numpy() == 0)[0], 1],
                'o', label='a=0', color='orange', mfc='none')
    plt.plot(z_tsne[np.where(condition.cpu().detach().numpy() == 1)[0], 0],
                z_tsne[np.where(condition.cpu().detach().numpy() == 1)[0], 1],
                '+', label='a=1', color='purple')
    plt.legend()
    plt.savefig('output/scatter_z_rep' + str(i) + '_iter' + str(i) + '.png')
    plt.tight_layout()
    plt.close()
   