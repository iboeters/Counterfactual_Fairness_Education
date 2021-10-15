import torch
import torch.nn as nn
# holds convolution functions; all functions without parameters; relu
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from torch.distributions import OneHotCategorical

# different classes:
# q(z)
# p(l|a, z)
# p(s|a)
# p(t|s, a, l, z)
# p(e|s, a, l, t, z)
# p(f|a, t, e, z) ; add later??
# p(y|a, f, z)

# h= n hidden layers
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

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out

# p(s|a)
class p_s_a_model(nn.Module):
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

    def forward(self, x, a):
        h1 = F.elu(self.lin1(x))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out


class CEVAE_model(nn.Module):
    def __init__(self, args, dim_a, dim_y, dim_s_con, dim_s_bin, dim_l, dim_t, dim_e, dim_f, dim_z, dim_q_h=100, dim_p_h=50):
        super().__init__()
        # Change structure in case of resolving variables
        self.q_z = q_z_yltef(dim_in=dim_y + dim_l + dim_t + dim_e + dim_f, dim_h=dim_q_h, dim_out=dim_z)
        self.p_l_az = p_l_az_model(dim_in=dim_a + dim_z, dim_h=dim_p_h, dim_out=dim_l)
        self.p_s_a = p_s_a_model(dim_in=dim_a, dim_h=dim_p_h, dim_out=dim_s_con + dim_s_bin)
        self.p_t_aslz = p_t_alsz_model(dim_in=dim_a + dim_s_con + dim_s_bin + dim_l + dim_z, dim_h=dim_p_h, dim_out=dim_t)
        self.p_e_altsz = p_e_altsz_model(dim_in=dim_a+dim_l+dim_t+dim_s_con+dim_s_bin+dim_z, dim_h=dim_p_h, dim_out=dim_e)
        self.p_y_afz = p_y_afz_model(dim_in=dim_a + dim_f + dim_z, dim_h=dim_p_h, dim_out=dim_y)
        self.p_z_dist = normal.Normal(torch.zeros(args.zDim).to(args.device), torch.ones(args.zDim).to(args.device))

    def forward(self, n_samples_z, device, batch_data, loss_dict, cat_bin_dict, eval=False, reconstruct=False, switch_a=False):
        y_batch, x_batch_con, x_batch_bin, r_batch, b_batch, a_batch = batch_data
        # INFER distribution over z
        # torch.cat: concatenate tensors: over dimension 1 --> per participant
        # q(z|a, b, x, r)
        z_infer = self.q_z.forward(observations=torch.cat((x_batch_con, x_batch_bin, r_batch, b_batch, a_batch), 1))


        #call other forward functions
        #calculate reconstruction loss
        #calculate y loss
        #save loss

        return loss_dict
