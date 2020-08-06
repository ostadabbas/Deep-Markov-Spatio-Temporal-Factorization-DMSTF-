
import numpy as np
import argparse
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable           
import os
import time
import sys
'Unloading matplotlib to load it later with Agg backend'
modules = []
for module in sys.modules:
    if module.startswith('matplotlib'):
        modules.append(module)
for module in modules:
    sys.modules.pop(module)
'##############'
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))
import pdb
from tqdm import tqdm
from generate_results_traffic import plot_result


class TemporalFactors(nn.Module):
    """
    Parameterizes the Gaussian weight p(w_t | z_t)
    """
    def __init__(self, factor_dim, z_dim, emission_dim):
        super(TemporalFactors, self).__init__()
        # initialize linear transformations used in the neural network
        self.lin_mean_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_mean_hidden_to_hidden = nn.Linear(emission_dim, 2*emission_dim)
        self.lin_mean_hidden_to_weight_loc = nn.Linear(2*emission_dim, 2*factor_dim)
        self.lin_sigma = nn.Linear(factor_dim, factor_dim) ## unused
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU() # nn.ReLU()
        
    def forward(self, z_t):
        """
        Given the latent z_t corresponding to the time step t
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(w_t | z_t)
        """
        
        # compute the 'weight mean' given z_t
        _weight_mean = self.relu(self.lin_mean_z_to_hidden(z_t))
        weight_mean = self.relu(self.lin_mean_hidden_to_hidden(_weight_mean))
        weight_params = self.lin_mean_hidden_to_weight_loc(weight_mean)
        
        # return loc, scale of w_t respectively, which can be fed into Normal
        return weight_params[:,0:factor_dim], weight_params[:,factor_dim:]
    
    
class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """
    def __init__(self, z_dim, u_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim + u_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim + u_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim + u_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim, z_dim + u_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU() # nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t_1, u_t_1):
        """
        Given the latent z_{t-1} and stimuli u_{t-1} corresponding to the time
        step t-1, we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1}, u_{t-1})
        """
        # stack z and u in a single vector if u is available
        if u_t_1 is not None:
            zu_t_1 = torch.cat((z_t_1, u_t_1), dim = 1)
        else:
            zu_t_1 = z_t_1
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(zu_t_1))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(zu_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        z_loc = (1 - gate) * self.lin_z_to_loc(zu_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input
        z_scale = self.lin_sig(self.relu(proposed_mean))
        # return loc, scale which can be fed into Normal
        return z_loc, z_scale
    

class SpatialFactors(nn.Module):
    """
    Parameterizes the RBF spatial factors  p(F | z_F)
    """
    def __init__(self, D, factor_dim, zF_dim):
        super(SpatialFactors, self).__init__()
        # initialize the linear transformations used in the neural network
        # shared structure
        self.lin_zF_to_hidden_0 = nn.Linear(zF_dim, 2*zF_dim)
        self.lin_hidden_0_to_hidden_1 = nn.Linear(2*zF_dim, 4*zF_dim)
        # mean and sigma for factor location
        self.lin_mean_hidden_to_factor_loc = nn.Linear(4*zF_dim, 2 * D * factor_dim)
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU() # nn.ReLU()
        
    def forward(self, z_F):
        """
        Given the latent z_F corresponding to spatial factor embedding
        we return the mean and sigma vectors that parameterize the
        (diagonal) gaussian distribution p(F | z_F) for factor location 
        and scale.
        """
        # computations for shared structure 
        _hidden_output = self.relu(self.lin_zF_to_hidden_0(z_F))
        hidden_output = self.relu(self.lin_hidden_0_to_hidden_1(_hidden_output))
        # compute the 'mean' and 'sigma' for factor location given z_F
        factor_loc_params = self.lin_mean_hidden_to_factor_loc(hidden_output)
        # return means, sigmas of factor loc, scale which can be fed into Normal
        return factor_loc_params[:,:D*factor_dim], factor_loc_params[:,D*factor_dim:]
    

class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, w_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on w_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, u_dim, rnn_dim):
        super(Combiner, self).__init__()
        # initialize the linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim + u_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the non-linearities used in the neural network
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, u_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(w_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, w_{t:T})
        """
        # stack z and u in a single vector if u is available
        if u_t_1 is not None:
            zu_t_1 = torch.cat((z_t_1, u_t_1), dim = 1)
        else:
            zu_t_1 = z_t_1
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(zu_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.lin_hidden_to_scale(h_combined)
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMFA(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution for the Deep Markov Factor Analysis
    """
    def __init__(self, n_data=100, T = 18,  factor_dim=10, z_dim=5,
                 emission_dim=5, u_dim=0,
                 transition_dim=5, zF_dim=5, n_class=1, sigma_obs = 0,
                 use_cuda=False,
                 rnn_dim = None, rnn_dropout_rate = 0.0, D = 300):
        super(DMFA, self).__init__()
        
        self.rnn_dim = rnn_dim
        # observation noise
        self.sig_obs = sigma_obs
        # instantiate pytorch modules used in the model and guide below
        self.temp = TemporalFactors(factor_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, u_dim, transition_dim)
        self.spat = SpatialFactors(D,factor_dim, zF_dim)
        
        if rnn_dim is not None: # initialize extended DMFA
            self.combiner = Combiner(z_dim, u_dim, rnn_dim*2)
            self.rnn = nn.RNN(input_size=factor_dim, hidden_size=rnn_dim,
                              nonlinearity='relu', batch_first=True,
                              bidirectional=True, num_layers=1, dropout=rnn_dropout_rate)
            # define a (trainable) parameter for the initial hidden state of the rnn
            self.h_0 = nn.Parameter(torch.zeros(2, 1, rnn_dim))
        
        """
        # define uniform pior p(c)
        """
        self.softmax = nn.Softmax(dim = 0)
        self.p_c = self.softmax(nn.Parameter(torch.ones(n_class)))
        """
        # define Gaussian pior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim).log()
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        
        self.z_0_mu = nn.Parameter(torch.rand(n_class, z_dim)- 1/2)
        
        init_sig = (torch.ones(n_class, z_dim) / (2 * n_class)* 0.15 * 5).log()
        self.z_0_sig = nn.Parameter(init_sig)
        """
        # define trainable parameters that help define
        # the probability distributions for inference
        # q(c), q(z_0)...q(z_T), q(w_1)...q(w_T), q(z_F), q(F_loc), q(F_scale)
        """
        
        self.q_c = torch.ones(n_data, n_class) / n_class
        self.q_z_0_mu = nn.Parameter(torch.rand(n_data, z_dim)- 1/2)
        init_sig = (torch.ones(n_data, z_dim) / (2 * n_class)*0.1).log()
        self.q_z_0_sig = nn.Parameter(init_sig)
        if rnn_dim is not None:
            self.q_z_mu = torch.zeros(n_data, T-1, z_dim)
            self.q_z_sig = torch.ones(n_data, T-1, z_dim)
        else:
            self.q_z_mu = nn.Parameter(torch.rand(n_data, T-1, z_dim)- 1/2)
            init_sig = (torch.ones(n_data, T-1, z_dim) / (2 * n_class) * 0.1).log()
            self.q_z_sig = nn.Parameter(init_sig)
        self.q_w_mu = nn.Parameter(torch.rand(n_data, T, factor_dim)- 1/2)
        init_sig = (torch.ones(n_data, T, factor_dim) / (2 * n_class) * 0.1).log()
        self.q_w_sig = nn.Parameter(init_sig)
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        init_sig = torch.ones(zF_dim).log()
        self.q_z_F_sig = nn.Parameter(init_sig)
        self.q_F_loc_mu = nn.Parameter(torch.rand(factor_dim, D)- 1/2)
        init_sig = (torch.ones(factor_dim, D) / (2 * factor_dim) * 0.1).log()
        self.q_F_loc_sig = nn.Parameter(init_sig)
        
        self.use_cuda = use_cuda
        # if on gpu cuda-ize all pytorch (sub)modules
        if use_cuda:
            self.cuda()        
        
    def Reparam(self, mu_latent, sigma_latent):
        eps = Variable(mu_latent.data.new(mu_latent.size()).normal_())
        return eps.mul(sigma_latent.exp()).add_(mu_latent)
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def model(self, u_values, z_values, w_values, 
              F_loc_values, zF_values):
        # data_points = number of data points in mini-batch
        # class_idxs = (data_points, 1)
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + 1, z_dim)
        # w_values = (data_points, time_points, factor_dim)
        # F_loc_values = (data_points, factor_dim, 3)
        # F_scale_values = (data_points, factor_dim)
        # zF_values = (data_points, zF_dim)
        
        # the number of data points and time steps
        # we need to process in the mini-batch
        N_max = w_values.size(0)
        T_max = w_values.size(1)
        
        # p(c) = Uniform(n_class)
        p_cs = self.p_c.repeat(N_max, 1)
        
        # p(z_0|c) = Normal(z_0_mu, I)
        p_z_0_mu = self.z_0_mu.repeat(N_max, 1, 1)
        p_z_0_sig = self.z_0_sig.repeat(N_max, 1, 1)
        
        # p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale)
        z_t_1 = z_values[:,:-1,:].reshape(N_max * (T_max-1), -1)
        if u_values is not None:
            u_t_1 = u_values[:,:-1,:].reshape(N_max * (T_max-1), -1)
        else:
            u_t_1 = None
        p_z_mu, p_z_sig = self.trans(z_t_1, u_t_1)
        p_z_mu = p_z_mu.view(N_max, T_max-1, -1)
        p_z_sig = p_z_sig.view(N_max, T_max-1, -1)
            
        # p(w_t|z_t) = Normal(w_loc, w_scale)
        z_t = z_values.reshape(N_max * T_max, -1)
        p_w_mu, p_w_sig = self.temp(z_t)
        p_w_mu = p_w_mu.view(N_max, T_max, -1)
        p_w_sig = p_w_sig.view(N_max, T_max, -1)
        
        # p(F_mu|z_F) = Normal(F_mu_loc, F_mu_scale)
        # p(F_sig|z_F) = Normal(F_sig_loc, F_sig_scale)
        p_F_loc_mu, p_F_loc_sig = self.spat(zF_values)
        p_F_loc_mu = p_F_loc_mu.view(factor_dim, -1)
        p_F_loc_sig = p_F_loc_sig.view(factor_dim, -1)
        
        # p(y|w,F) = Normal(w*f(F), sigma)
        # w : (data_points, time_points, factor_dim)
        # f(F): (data_points, factor_dim, voxel_num)
        f_F = F_loc_values.repeat(N_max, 1,1)
        y_hat_nn = torch.matmul(z_values, f_F)
        obs_noise = Variable(y_hat_nn.data.new(y_hat_nn.size()).normal_())
        y_hat = obs_noise.mul(self.sig_obs).add_(y_hat_nn)
        
        return p_z_0_mu, p_z_0_sig,\
                p_z_mu, p_z_sig,\
                p_w_mu, p_w_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                self.p_z_F_mu, self.p_z_F_sig,\
                y_hat,\
                p_cs
                
    # the guide q(w_{n,t})q(z_{n,t})p(z_{n,0})q(c_n)q(F)q(z_F) 
    # (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_idxs):
        
        # data_points = number of data points in mini-batch
        # mini_batch : (data_points, time_points, voxels)
        # mini_batch_idxs : indices of data points
        
        # this is the number of data points we need to process in the mini-batch
        N_max = mini_batch.size(0)
        
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs]
        q_z_0_sigs = self.q_z_0_sig[mini_batch_idxs]
        if self.rnn_dim is not None: # to be determrined later in code
            q_z_mus = torch.Tensor([]).reshape(N_max,0,q_z_0_mus.size(1))
            q_z_sigs = torch.Tensor([]).reshape(N_max,0,q_z_0_sigs.size(1))
        else:
            q_z_mus = self.q_z_mu[mini_batch_idxs]
            q_z_sigs = self.q_z_sig[mini_batch_idxs]        
        q_w_mus = self.q_w_mu[mini_batch_idxs] 
        q_w_sigs = self.q_w_sig[mini_batch_idxs]
        q_F_loc_mus = self.q_F_loc_mu
        q_F_loc_sigs = self.q_F_loc_sig
        q_z_F_mus = self.q_z_F_mu.view(1,-1)
        q_z_F_sigs = self.q_z_F_sig.view(1,-1)
        
        return q_z_0_mus,\
                q_z_0_sigs,\
                q_z_mus, q_z_sigs,\
                q_w_mus, q_w_sigs,\
                q_F_loc_mus, q_F_loc_sigs,\
                q_z_F_mus, q_z_F_sigs
                
    def forward(self, mini_batch, u_values, mini_batch_idxs):
        
        # get outputs from both modules: guide and model
        q_z_0_mus,\
        q_z_0_sigs,\
        q_z_mus, q_z_sigs,\
        q_w_mus, q_w_sigs,\
        q_F_loc_mus, q_F_loc_sigs,\
        q_z_F_mus, q_z_F_sigs = self.guide(mini_batch, mini_batch_idxs)
        
        w_values = self.Reparam(q_w_mus, q_w_sigs)
        z_0_values = self.Reparam(q_z_0_mus, q_z_0_sigs).unsqueeze(1)
        
        if self.rnn_dim is not None:
            # if on gpu we need the fully broadcast view of the rnn initial state
            # to be in contiguous gpu memory
            h_0_contig = self.h_0.expand(2, mini_batch.size(1),
                                         self.rnn.hidden_size).contiguous()
            rnn_output, _= self.rnn(w_values.permute(1,0,2), h_0_contig)
            z_t_values = torch.Tensor([]).reshape(z_0_values.size(0),0,z_0_values.size(2))
            z_prev = z_0_values.squeeze(1).clone()
            for i in range(1,w_values.size(1)):
                if u_values is not None:
                    u_prev = u_values[:,i-1,:]
                else:
                    u_prev = None
                loc, scale = self.combiner(z_prev, u_prev, rnn_output[i])
                z_prev = self.Reparam(loc, scale)
                z_t_values = torch.cat((z_t_values,z_prev.unsqueeze(1)), dim=1)
                q_z_mus = torch.cat((q_z_mus,loc.unsqueeze(1)), dim=1)
                q_z_sigs = torch.cat((q_z_sigs,scale.unsqueeze(1)), dim=1)
            self.q_z_mu[mini_batch_idxs] = q_z_mus
            self.q_z_sig[mini_batch_idxs] = q_z_sigs
        else:
            z_t_values = self.Reparam(q_z_mus, q_z_sigs)
        z_values = torch.cat((z_0_values, z_t_values), dim = 1)
        
        F_loc_values = self.Reparam(q_F_loc_mus, q_F_loc_sigs)

        zF_values = self.Reparam(q_z_F_mus, q_z_F_sigs)
        
        p_z_0_mu, p_z_0_sig,\
        p_z_mu, p_z_sig,\
        p_w_mu, p_w_sig,\
        p_F_loc_mu, p_F_loc_sig,\
        ps_z_F_mu, ps_z_F_sig,\
        y_hat,\
        p_cs = self.model(u_values, z_values, w_values, 
                          F_loc_values, zF_values)
        
        # this is the number of data points we need to process in the mini-batch    
        N_max = mini_batch.size(0)
        n_class = self.p_c.size(-1)
        
        """compute q_c based on equation 16 in https://arxiv.org/pdf/1611.05148.pdf"""
        z_0_vals = z_0_values.squeeze(1).repeat(1, n_class).view(N_max, n_class, -1)
        q_cs_Unlog = p_cs.log() \
                     - 1 / 2 * ((z_0_vals - p_z_0_mu) / (p_z_0_sig.exp()+1e-4)).pow(2).sum(dim = -1) \
                    - p_z_0_sig.sum(dim = -1)
        q_cs_log = q_cs_Unlog - q_cs_Unlog.logsumexp(dim = -1).view(-1,1).repeat(1, n_class)
        q_cs = q_cs_log.exp()
        """End"""
        
        self.q_c[mini_batch_idxs,:] = q_cs
        qs_z_0_mus = q_z_0_mus.repeat(1, n_class).view(N_max, n_class, -1)
        qs_z_0_sigs = q_z_0_sigs.repeat(1, n_class).view(N_max, n_class, -1)
        qs_F_loc_mu = self.q_F_loc_mu
        qs_F_loc_sig = self.q_F_loc_sig
        qs_z_F_mu = self.q_z_F_mu
        qs_z_F_sig = self.q_z_F_sig
        
        return y_hat,\
                q_cs, p_cs,\
                qs_z_0_mus, qs_z_0_sigs,\
                p_z_0_mu, p_z_0_sig,\
                q_z_mus, q_z_sigs,\
                p_z_mu, p_z_sig,\
                q_w_mus, q_w_sigs,\
                p_w_mu, p_w_sig,\
                qs_F_loc_mu, qs_F_loc_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                qs_z_F_mu, qs_z_F_sig,\
                ps_z_F_mu, ps_z_F_sig
                
    def restore_nonparameter_variables(self, u_values):
        # update q_z (without loading entire data) in case of using rnn
        z_0_values = self.Reparam(self.q_z_0_mu, self.q_z_0_sig).unsqueeze(1)
        if self.rnn_dim is not None:
            w_values = self.Reparam(self.q_w_mu, self.q_w_sig)        
            h_0_contig = self.h_0.expand(2, w_values.size(1),
                                         self.rnn.hidden_size).contiguous()
            rnn_output, _= self.rnn(w_values.permute(1,0,2), h_0_contig)
            z_prev = z_0_values.squeeze(1).clone()
            for i in range(1, w_values.size(1)):
                if u_values is not None:
                    u_prev = u_values[:,i-1,:]
                else:
                    u_prev = None
                loc, scale = self.combiner(z_prev, u_prev, rnn_output[i])
                z_prev = self.Reparam(loc, scale)
                self.q_z_mu[:, i-1, :] = loc
                self.q_z_sig[:, i-1, :] = scale
        # update q_c (without loading entire data)
        n_data = self.q_c.size(0)
        n_class = self.q_c.size(-1)
        p_cs = self.p_c.repeat(n_data, 1)
        p_z_0_mu = self.z_0_mu.repeat(n_data, 1, 1)
        p_z_0_sig = self.z_0_sig.repeat(n_data, 1, 1)
        z_0_vals = z_0_values.squeeze(1).repeat(1, n_class).view(n_data, n_class, -1)
        q_cs_Unlog = p_cs.log() \
                     - 1 / 2 * ((z_0_vals - p_z_0_mu) / (p_z_0_sig.exp()+1e-4)).pow(2).sum(dim = -1) \
                    - p_z_0_sig.sum(dim = -1)
        q_cs_log = q_cs_Unlog - q_cs_Unlog.logsumexp(dim = -1).view(-1,1).repeat(1, n_class)
        self.q_c = q_cs_log.exp()

def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    KLD = 1/2 * ( 2 * (p_sigma - q_sigma) 
                    - 1
                    + ((q_sigma.exp())/(p_sigma.exp()+1e-6)).pow(2)
                    + ( (p_mu - q_mu) / (p_sigma.exp()+1e-6) ).pow(2) )
    return KLD.sum(dim = -1)

def KLD_Cat(q, p):
    # sum (q log (q/p) )
    KLD = q * ((q+1e-4) / (p+1e-4)).log()
    return KLD.sum(dim = -1)

mse_loss = torch.nn.MSELoss(size_average=False, reduce=True)

def ELBO_Loss(mini_batch, y_hat, 
              q_cs, p_cs,
              qs_z_0_mus, qs_z_0_sigs,
              p_z_0_mu, p_z_0_sig,
              q_z_mus, q_z_sigs,
              p_z_mu, p_z_sig,
              q_w_mus, q_w_sigs,
              p_w_mu, p_w_sig,
              qs_F_loc_mu, qs_F_loc_sig,
              p_F_loc_mu, p_F_loc_sig,
              qs_z_F_mu, qs_z_F_sig,
              ps_z_F_mu, ps_z_F_sig, 
              annealing_factor = 1):
    
    'Annealing'
    'https://www.aclweb.org/anthology/K16-1002'
    # mini_batch : (data_points, time_points, voxels)
    
    rec_loss = mse_loss(y_hat, mini_batch)
    KL_c = KLD_Cat(q_cs, p_cs).sum()
    KL_z_0 = (q_cs *
                KLD_Gaussian(qs_z_0_mus, qs_z_0_sigs,
                             p_z_0_mu, p_z_0_sig)).sum()
    KL_z = KLD_Gaussian(q_z_mus, q_z_sigs, 
                        p_z_mu, p_z_sig).sum()
    KL_w = KLD_Gaussian(q_w_mus, q_w_sigs, 
                        p_w_mu, p_w_sig).sum()
    KL_F_loc = KLD_Gaussian(qs_F_loc_mu, qs_F_loc_sig,
                            p_F_loc_mu, p_F_loc_sig).sum()
    KL_z_F = KLD_Gaussian(qs_z_F_mu, qs_z_F_sig,
                          ps_z_F_mu, ps_z_F_sig)
    beta = annealing_factor
    
    return rec_loss + beta * (KL_c + KL_z_0 + KL_z)# + KL_w)# + KL_F_loc + KL_z_F)




# parse command-line arguments and execute the main method
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-t', '--T', type=int, default=6)
    parser.add_argument('-k', '--factor_dim', type=int, default=100)
    parser.add_argument('-de', '--emission_dim', type=int, default=15)
    parser.add_argument('-du', '--u_dim', type=int, default=0)
    parser.add_argument('-dzf', '--zF_dim', type=int, default=2)
    parser.add_argument('-c', '--n_class', type=int, default=1)
    parser.add_argument('-so', '--sigma_obs', type=float, default=0)
    parser.add_argument('-restore', action="store_true", default=False)
    parser.add_argument('-resume', action="store_true", default=False)
    parser.add_argument('-predict', action="store_true", default=False)
    parser.add_argument('-strain', action="store_false", default=True,
                        help = "whether to save train results every 50 epochs")
    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-days', type=int, default=5)
    parser.add_argument('-ID', type=str, default=None)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-drnn', '--rnn_dim', type=int, default=None)
    parser.add_argument('-file', '--data_file', type=str, default='./data_traffic/tensor.mat')
    parser.add_argument('-smod', '--model_path', type=str, default='./ckpt_files/')
    parser.add_argument('-dpath', '--dump_path', type=str, default='./')
    args = parser.parse_args()

    'Code starts Here'
    # we have N number of data points each of size (T*D)
    # we have N number of u each of size (T*u_dim)
    # we specify a vector with values 0, 1, ..., N-1
    # each datapoint for shuffling is a tuple ((T*D), (T*u_dim), n)
    
    # setting hyperparametrs
    
    T = args.T # number of time points in each sequence
    factor_dim = args.factor_dim # number of Gaussian blobs (spatial factors)
    emission_dim = args.emission_dim # hidden units from z to w (z_dim to factor_dim)
    u_dim = args.u_dim # dimension of stimuli embedding
    zF_dim = args.zF_dim # dimension of spatial factor embedding
    n_class = args.n_class # number of major clusters
    sigma_obs = args.sigma_obs # standard deviation of observation noise
    T_A = 100 # annealing iterations <= epoch_num
    use_cuda = False # set to True if using gpu
    Restore = args.restore # set to True if already trained
    predict = args.predict
    load_model = args.resume
    batch_size = args.batch_size # batch size
    epoch_num = args.epoch # number of epochs
    num_workers = 0 # number of workers to process dataset
    lr = args.lr # learning rate for adam optimizer
    z_dim = args.factor_dim # dimension of temporal latent variable z
    transition_dim = args.factor_dim * 2 # hidden units from z_{t-1} to z_t
    if args.rnn_dim is not None: # whether to add rnn extension to files/folders
        rnn_ext = '_rnn'
    else:
        rnn_ext = ''
    # dataset parameters
    root_dir = args.data_file
    # Path parameters
    save_PATH = args.model_path
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
    fig_PATH = args.dump_path + 'fig_files%s/' %(rnn_ext)
    if not os.path.exists(fig_PATH):
        os.makedirs(fig_PATH)
    PATH_DMFA = save_PATH + 'DMFA%s' %(rnn_ext)
    
    """
    DMFA SETUP & Training
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    """
    
    if root_dir[-3:] == 'txt':
        import pandas as pd
        data_st = pd.read_csv(root_dir, index_col = 0)
        data_st = data_st.values.reshape(-1,28,288).transpose(1,2,0).astype('float')
    if root_dir[-3:] == 'mat':
        from scipy.io import loadmat
        data_st = loadmat(root_dir)['tensor'].transpose(1,2,0).astype('float')  
    if root_dir[-3:] == 'npz':
        data_st = np.load(root_dir)['arr_0'].transpose(1,2,0).astype('float')  
    
    # normalize data for training    
    idxs_n = np.where(data_st != 0)
    data_mean = data_st[idxs_n].mean()
    data_std = data_st[idxs_n].std()
    dataa = (data_st - data_mean)/data_std
    # extract data dimensions
    D = dataa.shape[-1]
    n_data = dataa.shape[0]
    #form data for training
    training_set = [(torch.FloatTensor(y),torch.zeros(0),torch.LongTensor([i])) for i, y in enumerate(dataa)]
    # exclude test-set from training 
    if predict:
        training_set_part =  training_set[-args.days:]
        load_model = True
    else:
        training_set_part =  training_set[:-args.days]
    # form classes--It's dummy for these set of experiments
    classes = torch.LongTensor(np.zeros(n_data))
    # initialize model
    dmfa = DMFA(n_data = n_data,
                T = T,
                factor_dim = factor_dim,
                z_dim = z_dim,
                emission_dim = emission_dim,
                u_dim = u_dim,
                transition_dim = transition_dim,
                zF_dim = zF_dim,
                n_class = n_class,
                sigma_obs = sigma_obs,
                use_cuda = use_cuda,
                rnn_dim = args.rnn_dim,
                D = D)
    # freeze model for prediction
    if predict:
        dmfa.q_F_loc_mu.requires_grad = False
        dmfa.q_F_loc_sig.requires_grad = False
        dmfa.q_z_F_mu.requires_grad = False
        dmfa.q_z_F_sig.requires_grad = False
        for param in dmfa.trans.parameters():
            param.requires_grad = False
        for param in dmfa.spat.parameters():
            param.requires_grad = False
        for param in dmfa.temp.parameters():
            param.requires_grad = False
            
    if Restore == False:
        # set path to save figure results during training
        fig_PATH_train = fig_PATH + 'figs_train/'
        if not os.path.exists(fig_PATH_train):
            os.makedirs(fig_PATH_train)
        
        optim_dmfa = optim.Adam(dmfa.parameters(), lr = lr)
        # number of parameters  
        total_params = sum(p.numel() for p in dmfa.parameters())
        learnable_params = sum(p.numel() for p in dmfa.parameters() if p.requires_grad)
        print('Total Number of Parameters: %d' % total_params)
        print('Learnable Parameters: %d' %learnable_params)
        
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': num_workers}
        train_loader = data.DataLoader(training_set_part, **params)
        
        print("Training...")
        if load_model:
            dmfa.load_state_dict(torch.load(PATH_DMFA,
                              map_location=lambda storage, loc: storage))
        for i in range(epoch_num):
            time_start = time.time()
            loss_value = 0.0
            for batch_indx, batch_data in enumerate(tqdm(train_loader)):
            # update DMFA
            
                mini_batch, u_vals, mini_batch_idxs = batch_data
                
                mini_batch = Variable(mini_batch)
                mini_batch_idxs = Variable(mini_batch_idxs.reshape(-1))
                
                mini_batch = mini_batch.to(device)
                mini_batch_idxs = mini_batch_idxs.to(device)
                
                if u_dim == 0:
                    u_vals = None
                else:
                    u_vals = Variable(u_vals)
                    u_vals = u_vals.to(device)
    
                y_hat,\
                q_cs, p_cs,\
                qs_z_0_mus, qs_z_0_sigs,\
                p_z_0_mu, p_z_0_sig,\
                q_z_mus, q_z_sigs,\
                p_z_mu, p_z_sig,\
                q_w_mus, q_w_sigs,\
                p_w_mu, p_w_sig,\
                qs_F_loc_mu, qs_F_loc_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                qs_z_F_mu, qs_z_F_sig,\
                ps_z_F_mu, ps_z_F_sig\
                = dmfa.forward(mini_batch, u_vals, mini_batch_idxs)
    
            # set gradients to zero in each iteration
                optim_dmfa.zero_grad()
            
            # computing loss
                # excluding missing locations
                idxs_nonzero = (torch.FloatTensor(data_st[mini_batch_idxs].reshape(-1,T,D)) != 0)
                annealing_factor = 0.001# min(1.0, 0.01 + i / T_A) # inverse temperature
                loss_dmfa = ELBO_Loss(mini_batch[idxs_nonzero],
                                      y_hat[idxs_nonzero], 
                                      q_cs, p_cs,
                                      qs_z_0_mus, qs_z_0_sigs,
                                      p_z_0_mu, p_z_0_sig,
                                      q_z_mus, q_z_sigs,
                                      p_z_mu, p_z_sig,
                                      q_w_mus, q_w_sigs,
                                      p_w_mu, p_w_sig,
                                      qs_F_loc_mu, qs_F_loc_sig,
                                      p_F_loc_mu, p_F_loc_sig,
                                      qs_z_F_mu, qs_z_F_sig,
                                      ps_z_F_mu, ps_z_F_sig,
                                      annealing_factor)
                
            # back propagation
                loss_dmfa.backward(retain_graph = True)
                'https://stackoverflow.com/questions/55268726/pytorch-why-does-preallocating-memory-cause-trying-to-backward-through-the-gr'
                'https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method'
            # update parameters
                optim_dmfa.step()
    
                loss_value += loss_dmfa.item()
            
            acc = torch.sum(dmfa.q_c.argmax(dim=1)==classes).float()/n_data
            time_end = time.time()
            print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
            print('====> Epoch: %d ELBO_Loss : %0.4f Acc: %0.2f'
                  % ((i + 1), loss_value / len(train_loader.dataset), acc))
    
            torch.save(dmfa.state_dict(), PATH_DMFA)
            
            #draw plots once per 10 epochs
            if args.strain and i % 50 == 0:
                plot_result(dmfa, classes, fig_PATH_train,
                            prefix = 'epoch{%.3d}_'%i,
                            ext = ".png", data_st = [dataa, data_mean, data_std],
                            days = args.days, predict = args.predict)
            
    if Restore:
        dmfa.load_state_dict(torch.load(PATH_DMFA,
                                        map_location=lambda storage, loc: storage))
        params = {'batch_size': n_data,
                  'shuffle': False,
                  'num_workers': 0}
        train_loader = data.DataLoader(training_set, **params)
        _, batch_data = next(iter(enumerate(train_loader)))
        _, u_vals, _ = batch_data
        
        if u_dim == 0:
            u_vals = None
        else:
            u_vals = Variable(u_vals)
            u_vals = u_vals.to(device)
        dmfa.restore_nonparameter_variables(u_vals)
        
        plot_result(dmfa, classes, fig_PATH,
                    data_st = [dataa, data_mean, data_std],
                    days = args.days, ID = args.ID)

"""
DMFA SETUP & Training--END
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""