# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:16:32 2019

@author: Amir
"""
import numpy as np
import argparse
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import os
from os.path import exists
import time
import pyro
from pyro.distributions import Categorical, Normal
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import matplotlib
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nib
#from control import matlab # "conda install -c conda-forge control"
#from torch.distributions.categorical import Categorical
# plt.subplots_adjust(top = 0.99, bottom=0.1, hspace=0.1, wspace=0.1)
#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))


## https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, root_dir):
        'Initialization'
        self.list_IDs = list_IDs
        self.root_dir = root_dir

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        y = np.load(os.path.join(self.root_dir, 'fMRI_%.5d.npz' %ID))['arr_0'].reshape(-1)
        s = np.load(os.path.join(self.root_dir, 'stimuli_%.5d.npz' %ID))['arr_0'].reshape(-1)
        #normalize
        y = y / 378  #std()
        y = torch.FloatTensor(y)
        s = torch.FloatTensor(s)
        i = torch.LongTensor([index])

        return (y,s,i)


class SyntheticGenerator:
    """
    Synthesize fMRI dataset
    """
    def __init__(self, 
                 n_data=100,
                 T = 10,
                 factor_dim=10,
                 z_dim=5,
                 u_dim=2,
                 n_class=5,
                 sigma_obs = 1e-2,
                 image_dims = (10,10,10), 
                 ratio_max_space = 0.2,
                 ratio_sig = 0.05):
        super(SyntheticGenerator, self).__init__()
        
        """ observation noise """
        self.sig_obs = sigma_obs
        self.ratio_sig = ratio_sig
        self.n_data = n_data
        self.T = T
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.tanh = nn.Tanh()
        """ 3D coordinates of voxels: # of voxels times 3"""
        v_i, v_j, v_k = image_dims
        self.voxl_locs = torch.FloatTensor([[i - v_i/2, j - v_j/2, k - v_k/2] 
                                            for i in range(v_i)
                                            for j in range(v_j)
                                            for k in range(v_k)])
        """
        # generate u_values
        """
        u_values = torch.zeros(T, u_dim)
        step = int(T / (u_dim + 1))
        for i in range(1, u_dim + 1):
            if i == u_dim:
                u_values[i * step:, i-1] = 1 
            else:
                u_values[i * step:(i+1) * step, i-1] = 1
        self.u_values = u_values.repeat(n_data, 1 , 1)
        """
        # generate a random p(c)
        """
        p_c = torch.rand(n_class)
        self.p_c = p_c / p_c.sum()
        """
        # p(z_0|c)
        """
        self.z_0_mu = torch.rand(n_class, z_dim) - 1/2
        self.z_0_sig = torch.ones(n_class, z_dim) / (2 * n_class)
        """
        # parameters for generating
        # p(z_t|z_{t-1}, u{t-1}), p(w_t|z_t), p(F_loc), p(F_scale)
        """
        sys = matlab.rss(states = z_dim, outputs = 1, inputs = u_dim)
        self.A_z = torch.FloatTensor(sys.A)
        self.B_z = torch.FloatTensor(sys.B)
        self.A_w = torch.randn(factor_dim, z_dim)
        
        self.F_loc_mu = (torch.rand(factor_dim, 3) - 1/2)\
                        * torch.FloatTensor(image_dims)
        self.F_loc_sig = torch.ones(factor_dim, 3)\
                         * torch.FloatTensor(image_dims)\
                         / (2 * max(factor_dim, 1/ratio_max_space) * 0.2)
        self.F_scale_mu = (torch.rand(factor_dim) + 1/2)\
                          * min(image_dims)\
                          / (2 * max(factor_dim, 1/ratio_max_space))
        self.F_scale_sig = (self.F_scale_mu * ratio_sig).mean()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        vox_num = self.voxl_locs.size(0)
        dim_0, dim_1, dim_2 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,1,vox_num).view(dim_0, dim_1, vox_num, dim_2)
        vox_locs = self.voxl_locs.repeat(dim_0, dim_1, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, dim_1, 1).repeat(1, 1, vox_num)
        return torch.exp(-torch.sum(torch.pow(vox_locs - F_loc_vals, 2), dim = 3) / (2 * F_scale_vals.pow(2)))
        
        
    def Reparam(self, mu_latent, sigma_latent):
        # std = logvar_z.mul(0.5).exp() 
        eps = mu_latent.data.new(mu_latent.size()).normal_()
        # eps = eps.cuda()
        return eps.mul(sigma_latent).add_(mu_latent)
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def model(self):
        # data_points = number of data points in mini-batch
        # class_idxs = (data_points, 1)
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + 1, z_dim)
        # w_values = (data_points, time_points, factor_dim)
        # F_loc_values = (data_points, factor_dim, 3)
        # F_scale_values = (data_points, factor_dim)
        # zF_values = (data_points, zF_dim)
        
        """ sample class numbers from p_c """
        classes = Categorical(self.p_c).sample((self.n_data,))
        
        """ sample p(z_0|c) """
        z_0_mus = self.z_0_mu[classes]
        z_0_sigs = self.z_0_sig[classes]
        z_0_values = self.Reparam(z_0_mus, z_0_sigs)
        # p(z_0|c) = Normal(z_0_mu, I)
        
        """ p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale) """
        """ p(w_t|z_t) = Normal(w_loc, w_scale) """
        z_values = torch.zeros(self.n_data, self.T, self.z_dim)
        w_values = torch.zeros(self.n_data, self.T, self.factor_dim)
        z_prev = z_0_values
        for i in range(self.T):
            z_mus = self.A_z.matmul(z_prev.t())\
                        + self.B_z.matmul(self.u_values[:,i,:].t())
            z_mus = self.tanh(z_mus)
            z_sigs = z_mus * self.ratio_sig
            z_prev = self.Reparam(z_mus.t(), z_sigs.t())
            z_values[:,i,:] = z_prev
            w_mus = self.A_w.matmul(z_prev.t())
            w_mus = self.tanh(w_mus)
            w_sigs = w_mus * self.ratio_sig
            w_values[:,i,:] = self.Reparam(w_mus.t(), w_sigs.t())
            
        F_loc_mus = self.F_loc_mu.repeat(self.n_data, 1, 1)
        F_loc_sigs = self.F_loc_sig.repeat(self.n_data, 1, 1)
        F_loc_values = self.Reparam(F_loc_mus, F_loc_sigs)
        F_scale_mus = self.F_scale_mu.repeat(self.n_data, 1)
        F_scale_sigs = self.F_scale_sig.repeat(self.n_data, self.factor_dim)
        F_scale_values = self.Reparam(F_scale_mus, F_scale_sigs)

        
        # p(y|w,F) = Normal(w*f(F), sigma)
        # w : (data_points, time_points, factor_dim)
        # f(F): (data_points, factor_dim, voxel_num)
        f_F = self.RBF_to_Voxel(F_loc_values, F_scale_values)
        y_nn = torch.matmul(w_values, f_F)
        obs_noise = y_nn.data.new(y_nn.size()).normal_()
        y = obs_noise.mul(self.sig_obs * y_nn).add_(y_nn)
        
        training_set = [(y[i], self.u_values[i], torch.LongTensor([i]))
                        for i in range(self.n_data)]
        
        return classes,\
                z_0_values,\
                z_values,\
                w_values,\
                self.F_loc_mu, self.F_loc_sig,\
                self.F_scale_mu, self.F_scale_sig,\
                self.z_0_mu, self.z_0_sig,\
                self.p_c,\
                self.u_values,\
                y,\
                training_set,\
                self.voxl_locs
    
    
class SyntheticNeuralGenerator:
    """
    Synthesize fMRI dataset
    """
    def __init__(self, 
                 n_data=100,
                 T = 10,
                 factor_dim=10,
                 z_dim=5,
                 u_dim=2,
                 emission_dim=5,
                 transition_dim=5,
                 n_class=5,
                 sigma_obs = 1e-2,
                 image_dims = (10,10,10), 
                 ratio_max_space = 0.2,
                 ratio_sig = 0.05):
        super(SyntheticNeuralGenerator, self).__init__()
        
        """ observation noise """
        self.sig_obs = sigma_obs
        self.ratio_sig = ratio_sig
        self.n_data = n_data
        self.T = T
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        """ instantiate pytorch modules"""
        self.temp = TemporalFactors(factor_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, u_dim, transition_dim)
        """ 3D coordinates of voxels: # of voxels times 3"""
        v_i, v_j, v_k = image_dims
        self.voxl_locs = torch.FloatTensor([[i - v_i/2, j - v_j/2, k - v_k/2] 
                                            for i in range(v_i)
                                            for j in range(v_j)
                                            for k in range(v_k)])
        """
        # generate u_values
        """
        u_values = torch.zeros(T, u_dim)
        step = int(T / (u_dim + 1))
        for i in range(1, u_dim + 1):
            if i == u_dim:
                u_values[i * step:, i-1] = 1 
            else:
                u_values[i * step:(i+1) * step, i-1] = 1
        self.u_values = u_values.repeat(n_data, 1 , 1)
        """
        # generate a random p(c)
        """
        p_c = torch.rand(n_class)
        self.p_c = p_c / p_c.sum()
        """
        # p(z_0|c)
        """
        self.z_0_mu = torch.rand(n_class, z_dim) - 1/2
        self.z_0_sig = torch.ones(n_class, z_dim) / (2 * n_class)
        """
        # parameters for generating
        # p(F_loc), p(F_scale)
        """
        
        self.F_loc_mu = (torch.rand(factor_dim, 3) - 1/2)\
                        * torch.FloatTensor(image_dims)
        self.F_loc_sig = torch.ones(factor_dim, 3)\
                         * torch.FloatTensor(image_dims)\
                         / (2 * max(factor_dim, 1/ratio_max_space) * 0.2)
        self.F_scale_mu = (torch.rand(factor_dim) + 1/2)\
                          * min(image_dims)\
                          / (2 * max(factor_dim, 1/ratio_max_space))
        self.F_scale_sig = (self.F_scale_mu * ratio_sig).mean()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        vox_num = self.voxl_locs.size(0)
        dim_0, dim_1, dim_2 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,1,vox_num).view(dim_0, dim_1, vox_num, dim_2)
        vox_locs = self.voxl_locs.repeat(dim_0, dim_1, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, dim_1, 1).repeat(1, 1, vox_num)
        return torch.exp(-torch.sum(torch.pow(vox_locs - F_loc_vals, 2), dim = 3) / (2 * F_scale_vals.pow(2)))
        
        
    def Reparam(self, mu_latent, sigma_latent):
        # std = logvar_z.mul(0.5).exp() 
        eps = mu_latent.data.new(mu_latent.size()).normal_()
        # eps = eps.cuda()
        return eps.mul(sigma_latent).add_(mu_latent)
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def model(self):
        # data_points = number of data points in mini-batch
        # class_idxs = (data_points, 1)
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + 1, z_dim)
        # w_values = (data_points, time_points, factor_dim)
        # F_loc_values = (data_points, factor_dim, 3)
        # F_scale_values = (data_points, factor_dim)
        # zF_values = (data_points, zF_dim)
        
        """ sample class numbers from p_c """
        classes = Categorical(self.p_c).sample((self.n_data,))
        
        """ sample p(z_0|c) """
        z_0_mus = self.z_0_mu[classes]
        z_0_sigs = self.z_0_sig[classes]
        z_0_values = self.Reparam(z_0_mus, z_0_sigs)
        # p(z_0|c) = Normal(z_0_mu, I)
        
        """ p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale) """
        """ p(w_t|z_t) = Normal(w_loc, w_scale) """
        z_values = torch.zeros(self.n_data, self.T, self.z_dim)
        w_values = torch.zeros(self.n_data, self.T, self.factor_dim)
        z_prev = z_0_values
        for i in range(self.T):
            z_mus, z_sigs = self.trans(z_prev, self.u_values[:,i,:])
            z_prev = self.Reparam(z_mus, z_sigs)
            z_values[:,i,:] = z_prev
            w_mus, w_sigs = self.temp(z_prev)
            w_values[:,i,:] = self.Reparam(w_mus, w_sigs)
            
        F_loc_mus = self.F_loc_mu.repeat(self.n_data, 1, 1)
        F_loc_sigs = self.F_loc_sig.repeat(self.n_data, 1, 1)
        F_loc_values = self.Reparam(F_loc_mus, F_loc_sigs)
        F_scale_mus = self.F_scale_mu.repeat(self.n_data, 1)
        F_scale_sigs = self.F_scale_sig.repeat(self.n_data, self.factor_dim)
        F_scale_values = self.Reparam(F_scale_mus, F_scale_sigs)

        
        # p(y|w,F) = Normal(w*f(F), sigma)
        # w : (data_points, time_points, factor_dim)
        # f(F): (data_points, factor_dim, voxel_num)
        f_F = self.RBF_to_Voxel(F_loc_values, F_scale_values)
        y_nn = torch.matmul(w_values, f_F)
        obs_noise = y_nn.data.new(y_nn.size()).normal_()
        y = obs_noise.mul(self.sig_obs * y_nn).add_(y_nn)
        
        training_set = [(y[i], self.u_values[i], torch.LongTensor([i]))
                        for i in range(self.n_data)]
        
        return classes,\
                z_0_values,\
                z_values,\
                w_values,\
                self.F_loc_mu, self.F_loc_sig,\
                self.F_scale_mu, self.F_scale_sig,\
                self.z_0_mu, self.z_0_sig,\
                self.p_c,\
                self.u_values,\
                y,\
                training_set,\
                self.voxl_locs



class TemporalFactors(nn.Module):
    """
    Parameterizes the Gaussian weight p(w_t | z_t)
    """
    def __init__(self, factor_dim, z_dim, emission_dim):
        super(TemporalFactors, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_mean_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_mean_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_mean_hidden_to_weight_loc = nn.Linear(emission_dim, factor_dim)
        self.lin_sigma = nn.Linear(factor_dim, factor_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU() # nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, z_t):
        """
        Given the latent z_t corresponding to the time step t
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(w_t | z_t)
        """
        
        # compute the 'weight mean' given z_t
        _weight_mean = self.relu(self.lin_mean_z_to_hidden(z_t))
        weight_mean = self.relu(self.lin_mean_hidden_to_hidden(_weight_mean))
        weight_loc = self.lin_mean_hidden_to_weight_loc(weight_mean)
        
        # compute the weight scale using the weight loc
        # from above as input. The softplus ensures that scale is positive
        weight_scale = self.softplus(self.lin_sigma(self.relu(weight_loc)))
        # return loc, scale of w_t which can be fed into Normal
        return weight_loc, weight_scale
    
    
class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """
    def __init__(self, z_dim, u_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
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
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU() # nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, u_t_1):
        """
        Given the latent z_{t-1} and stimuli u_{t-1} corresponding to the time
        step t-1, we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1}, u_{t-1})
        """
        # stack z and u in a single vector
        zu_t_1 = torch.cat((z_t_1, u_t_1), dim = 1)
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
        # mean from above as input. the softplus ensures that scale is positive
        z_scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return z_loc, z_scale
    

class SpatialFactors(nn.Module):
    """
    Parameterizes the RBF spatial factors  p(F | z_F)
    """
    def __init__(self, factor_dim, zF_dim):
        super(SpatialFactors, self).__init__()
        # initialize the six linear transformations used in the neural network
        # shared structure
        self.lin_zF_to_hidden_0 = nn.Linear(zF_dim, factor_dim)
        self.lin_hidden_0_to_hidden_1 = nn.Linear(factor_dim, 2 * factor_dim)
        # mean and sigma for factor location
        self.lin_mean_hidden_to_factor_loc = nn.Linear(2 * factor_dim, 3 * factor_dim)
        self.lin_sigma_factor_loc = nn.Linear(3 * factor_dim, 3 * factor_dim)
        # mean and sigma for factor scale
        self.lin_mean_hidden_to_factor_scale = nn.Linear(2 * factor_dim, factor_dim)
        self.lin_sigma_factor_scale = nn.Linear(factor_dim, 1)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU() # nn.ReLU()
        self.softplus = nn.Softplus()
        
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
        factor_loc_mean = self.lin_mean_hidden_to_factor_loc(hidden_output)
        factor_loc_sigma = self.softplus(self.lin_sigma_factor_loc(self.relu(factor_loc_mean)))
        # compute the 'mean' and 'sigma' for factor scale given z_F
        factor_scale_mean = self.softplus(self.lin_mean_hidden_to_factor_scale(hidden_output))
        factor_scale_sigma = self.softplus(self.lin_sigma_factor_scale(factor_scale_mean))
        # return means, sigmas of factor loc, scale which can be fed into Normal
        return factor_loc_mean, factor_loc_sigma, factor_scale_mean, factor_scale_sigma
    

class DMFA(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Factor Analysis
    """
    def __init__(self, n_data=100, T = 10,  factor_dim=10, z_dim=5,
                 emission_dim=5, u_dim=2,
                 transition_dim=5, zF_dim=5, n_class=5, sigma_obs = 1e-2, image_dims = None,
                 voxel_locations = None, use_cuda=False):
        super(DMFA, self).__init__()
        
        # observation noise
        self.sig_obs = sigma_obs
        # 3D coordinates of voxels: # of voxels times 3
        self.voxl_locs = voxel_locations
        # instantiate pytorch modules used in the model and guide below
        self.temp = TemporalFactors(factor_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, u_dim, transition_dim)
        self.spat = SpatialFactors(factor_dim, zF_dim)
        """
        # define uniform pior p(c)
        """
        self.smax = nn.Softmax(dim = 0)
        self.p_c = self.smax(nn.Parameter(torch.ones(n_class)))
        # self.p_c = torch.ones(n_class) / n_class
        """
        # define Gaussian pior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim)
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        
        self.z_0_mu = nn.Parameter(torch.rand(n_class, z_dim) - 1/2)
        
        # self.z_0_mu = nn.Parameter(torch.zeros(n_class, z_dim))
        self.z_0_sig = torch.ones(n_class, z_dim) / (2 * n_class) # drop the division if needed
        """
        # define trainable parameters that help define
        # the probability distributions for inference
        # q(c), q(z_0)...q(z_T), q(w_1)...q(w_T), q(z_F), q(F_loc), q(F_scale)
        """
        self.softmax = nn.Softmax(dim = 1)
        self.softplus = nn.Softplus()
        
        self.q_c = self.softmax(nn.Parameter(torch.rand(n_data, n_class)))
        self.q_z_0_mu = nn.Parameter(torch.rand(n_data, z_dim)- 1/2)
        self.q_z_0_sig = self.softplus(nn.Parameter(torch.ones(n_data, z_dim) / (2 * n_class)))
        self.q_z_mu = nn.Parameter(torch.rand(n_data, T, z_dim) - 1/2)
        self.q_z_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, z_dim) / (2 * n_class)))
        self.q_w_mu = nn.Parameter(torch.rand(n_data, T, factor_dim)- 1/2)
        self.q_w_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, factor_dim) / (2 * n_class)))
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        self.q_z_F_sig = self.softplus(nn.Parameter(torch.ones(zF_dim)))
        self.q_F_loc_mu = nn.Parameter((torch.rand(factor_dim, 3) - 1/2) 
                                        * torch.FloatTensor(image_dims))
        self.q_F_loc_sig = self.softplus(nn.Parameter(torch.ones(factor_dim, 3)
                                                        * torch.FloatTensor(image_dims)
                                                        / (2 * factor_dim * 0.2)))
        
        self.q_F_scale_mu = self.softplus(nn.Parameter((torch.rand(factor_dim) + 1/2)
                                                        * min(image_dims)
                                                        / (2 * factor_dim)))
        self.q_F_scale_sig = self.softplus(nn.Parameter(self.q_F_scale_mu.data.mean() * 0.05))
        
        
        """
        self.q_c = self.softmax(nn.Parameter(torch.ones(n_data, n_class)))
        self.q_z_0_mu = nn.Parameter(torch.zeros(n_data, z_dim))
        self.q_z_0_sig = self.softplus(nn.Parameter(torch.ones(n_data, z_dim)))
        self.q_z_mu = nn.Parameter(torch.zeros(n_data, T, z_dim))
        self.q_z_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, z_dim)))
        self.q_w_mu = nn.Parameter(torch.zeros(n_data, T, factor_dim))
        self.q_w_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, factor_dim)))
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        self.q_z_F_sig = self.softplus(nn.Parameter(torch.ones(zF_dim)))
        self.q_F_loc_mu = nn.Parameter(torch.zeros(factor_dim, 3))
        self.q_F_loc_sig = self.softplus(nn.Parameter(torch.ones(factor_dim, 3)))
        self.q_F_scale_mu = self.softplus(nn.Parameter(torch.ones(factor_dim)))
        self.q_F_scale_sig = self.softplus(nn.Parameter(torch.ones(1)))
        """
        
        self.use_cuda = use_cuda
        # if on gpu cuda-ize all pytorch (sub)modules
        if use_cuda:
            self.cuda()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        vox_num = self.voxl_locs.size(0)
        dim_0, dim_1, dim_2 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,1,vox_num).view(dim_0, dim_1, vox_num, dim_2)
        vox_locs = self.voxl_locs.repeat(dim_0, dim_1, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, dim_1, 1).repeat(1, 1, vox_num)
        return torch.exp(-torch.sum(torch.pow(vox_locs - F_loc_vals, 2), dim = 3) / (2 * F_scale_vals.pow(2)))
        
        
    def Reparam(self, mu_latent, sigma_latent):
        # std = logvar_z.mul(0.5).exp() 
        eps = Variable(mu_latent.data.new(mu_latent.size()).normal_())
        # eps = eps.cuda()
        return eps.mul(sigma_latent).add_(mu_latent)
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def model(self, u_values, z_values, w_values, 
              F_loc_values, F_scale_values, zF_values):
        # data_points = number of data points in mini-batch
        # class_idxs = (data_points, 1)
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + 1, z_dim)
        # w_values = (data_points, time_points, factor_dim)
        # F_loc_values = (data_points, factor_dim, 3)
        # F_scale_values = (data_points, factor_dim)
        # zF_values = (data_points, zF_dim)
        
        # this is the number of time steps we need to process in the mini-batch
        N_max = u_values.size(0)
        T_max = u_values.size(1)
        
        # p(c) = Uniform(n_class)
        p_cs = self.p_c.repeat(N_max, 1)
        
        # p(z_0|c) = Normal(z_0_mu, I)
        p_z_0_mu = self.z_0_mu.repeat(N_max, 1, 1)
        p_z_0_sig = self.z_0_sig.repeat(N_max, 1, 1)
        
        # p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale)
        z_t_1 = z_values[:,:-1,:].reshape(N_max * T_max, -1)
        u_t_1 = u_values.view(N_max * T_max, -1)
        p_z_mu, p_z_sig = self.trans(z_t_1, u_t_1)
        p_z_mu = p_z_mu.view(N_max, T_max, -1)
        p_z_sig = p_z_sig.view(N_max, T_max, -1)
            
        # p(w_t|z_t) = Normal(w_loc, w_scale)
        z_t = z_values[:,1:,:].reshape(N_max * T_max, -1)
        p_w_mu, p_w_sig = self.temp(z_t)
        p_w_mu = p_w_mu.view(N_max, T_max, -1)
        p_w_sig = p_w_sig.view(N_max, T_max, -1)
        
        # p(F_mu|z_F) = Normal(F_mu_loc, F_mu_scale)
        # p(F_sig|z_F) = Normal(F_sig_loc, F_sig_scale)
        p_F_loc_mu, p_F_loc_sig, p_F_scale_mu, p_F_scale_sig = self.spat(zF_values)
        p_F_loc_mu = p_F_loc_mu.view(N_max, -1, 3).mean(dim = 0)
        p_F_loc_sig = p_F_loc_sig.view(N_max, -1, 3).mean(dim = 0)
        p_F_scale_mu = p_F_scale_mu.mean(dim = 0).view(-1,1)
        p_F_scale_sig = p_F_scale_sig.mean(dim = 0).repeat(F_scale_values.size(-1)).view(-1,1)
        
        # p(y|w,F) = Normal(w*f(F), sigma)
        # w : (data_points, time_points, factor_dim)
        # f(F): (data_points, factor_dim, voxel_num)
        f_F = self.RBF_to_Voxel(F_loc_values, F_scale_values)
        y_hat_nn = torch.matmul(w_values, f_F)
        obs_noise = Variable(y_hat_nn.data.new(y_hat_nn.size()).normal_())
        y_hat = obs_noise.mul(self.sig_obs).add_(y_hat_nn)
        
        return p_z_0_mu, p_z_0_sig,\
                p_z_mu, p_z_sig,\
                p_w_mu, p_w_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                p_F_scale_mu, p_F_scale_sig,\
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
        
        q_cs = self.q_c[mini_batch_idxs]
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs]
        q_z_0_sigs = self.q_z_0_sig[mini_batch_idxs]
        q_z_mus = self.q_z_mu[mini_batch_idxs]
        q_z_sigs = self.q_z_sig[mini_batch_idxs]        
        q_w_mus = self.q_w_mu[mini_batch_idxs] 
        q_w_sigs = self.q_w_sig[mini_batch_idxs]        
        q_F_loc_mus = self.q_F_loc_mu.repeat(N_max, 1, 1)
        q_F_loc_sigs = self.q_F_loc_sig.repeat(N_max, 1, 1)
        q_F_scale_mus = self.q_F_scale_mu.repeat(N_max, 1)
        q_F_scale_sigs = self.q_F_scale_sig.repeat(q_F_scale_mus.size())
        q_z_F_mus = self.q_z_F_mu.repeat(N_max, 1)
        q_z_F_sigs = self.q_z_F_sig.repeat(N_max, 1)
        
        return q_cs,\
                q_z_0_mus,\
                q_z_0_sigs,\
                q_z_mus, q_z_sigs,\
                q_w_mus, q_w_sigs,\
                q_F_loc_mus, q_F_loc_sigs,\
                q_F_scale_mus,\
                q_F_scale_sigs,\
                q_z_F_mus, q_z_F_sigs
                
    def forward(self, mini_batch, u_values, mini_batch_idxs):
        
        # get outputs from both modules: guide and model
        q_cs,\
        q_z_0_mus,\
        q_z_0_sigs,\
        q_z_mus, q_z_sigs,\
        q_w_mus, q_w_sigs,\
        q_F_loc_mus, q_F_loc_sigs,\
        q_F_scale_mus,\
        q_F_scale_sigs,\
        q_z_F_mus, q_z_F_sigs = self.guide(mini_batch, mini_batch_idxs)
        
        z_0_values = self.Reparam(q_z_0_mus, q_z_0_sigs).unsqueeze(1)
        z_t_values = self.Reparam(q_z_mus, q_z_sigs)
        z_values = torch.cat((z_0_values, z_t_values), dim = 1)
        w_values = self.Reparam(q_w_mus, q_w_sigs)
        F_loc_values = self.Reparam(q_F_loc_mus, q_F_loc_sigs)
        F_scale_values = self.Reparam(q_F_scale_mus, q_F_scale_sigs)
        F_scale_values = torch.clamp(F_scale_values, min = 1e-1)
        zF_values = self.Reparam(q_z_F_mus, q_z_F_sigs)
        
        p_z_0_mu, p_z_0_sig,\
        p_z_mu, p_z_sig,\
        p_w_mu, p_w_sig,\
        p_F_loc_mu, p_F_loc_sig,\
        p_F_scale_mu, p_F_scale_sig,\
        ps_z_F_mu, ps_z_F_sig,\
        y_hat,\
        p_cs = self.model(u_values, z_values, w_values, 
                          F_loc_values, F_scale_values, zF_values)
              
        N_max = mini_batch.size(0)
        n_class = self.q_c.size(-1)
        
        qs_z_0_mus = q_z_0_mus.repeat(1, n_class).view(N_max, n_class, -1)
        qs_z_0_sigs = q_z_0_sigs.repeat(1, n_class).view(N_max, n_class, -1)
        qs_F_loc_mu = self.q_F_loc_mu
        qs_F_loc_sig = self.q_F_loc_sig
        qs_F_scale_mu = self.q_F_scale_mu.view(-1,1)
        qs_F_scale_sig = self.q_F_scale_sig.repeat(self.q_F_scale_mu.size()).view(-1,1)
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
                qs_F_scale_mu, qs_F_scale_sig,\
                p_F_scale_mu, p_F_scale_sig,\
                qs_z_F_mu, qs_z_F_sig,\
                ps_z_F_mu, ps_z_F_sig
                
     



           
                
"""
PYRO Module
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""
                
class PDMFA(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Factor Analysis
    """
    def __init__(self, n_data=100, batch_size = 5, T = 10,  factor_dim=10, z_dim=5,
                 emission_dim=5, u_dim=2,
                 transition_dim=5, zF_dim=5, n_class=5, sigma_obs = 1e-2, image_dims = None,
                 voxel_locations = None, use_cuda=False):
        super(PDMFA, self).__init__()
        
        self.T = T
        self.factor_dim = factor_dim
        self.batch_size = batch_size
        # observation noise
        self.sig_obs = sigma_obs
        # 3D coordinates of voxels: # of voxels times 3
        self.voxl_locs = voxel_locations
        # instantiate pytorch modules used in the model and guide below
        self.temp = TemporalFactors(factor_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, u_dim, transition_dim)
        self.spat = SpatialFactors(factor_dim, zF_dim)
        """
        # define uniform pior p(c)
        """
        self.smax = nn.Softmax(dim = 0)
        self.p_c = self.smax(nn.Parameter(torch.ones(n_class)))
        # self.p_c = torch.ones(n_class) / n_class
        """
        # define Gaussian pior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim)
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        
        self.z_0_mu = nn.Parameter(torch.rand(n_class, z_dim) - 1/2)
        
        # self.z_0_mu = nn.Parameter(torch.zeros(n_class, z_dim))
        self.z_0_sig = torch.ones(n_class, z_dim) / (2 * n_class) # drop the division if needed
        """
        # define trainable parameters that help define
        # the probability distributions for inference
        # q(c), q(z_0)...q(z_T), q(w_1)...q(w_T), q(z_F), q(F_loc), q(F_scale)
        """
        self.softmax = nn.Softmax(dim = 1)
        self.softplus = nn.Softplus()
        
        self.q_c = self.softmax(nn.Parameter(torch.rand(n_data, n_class)))
        self.q_z_0_mu = nn.Parameter(torch.rand(n_data, z_dim)- 1/2)
        self.q_z_0_sig = self.softplus(nn.Parameter(torch.ones(n_data, z_dim) / (2 * n_class)))
        self.q_z_mu = nn.Parameter(torch.rand(n_data, T, z_dim) - 1/2)
        self.q_z_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, z_dim) / (2 * n_class)))
        self.q_w_mu = nn.Parameter(torch.rand(n_data, T, factor_dim)- 1/2)
        self.q_w_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, factor_dim) / (2 * n_class)))
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        self.q_z_F_sig = self.softplus(nn.Parameter(torch.ones(zF_dim)))
        self.q_F_loc_mu = nn.Parameter((torch.rand(factor_dim, 3) - 1/2) 
                                        * torch.FloatTensor(image_dims))
        self.q_F_loc_sig = self.softplus(nn.Parameter(torch.ones(factor_dim, 3)
                                                        * torch.FloatTensor(image_dims)
                                                        / (2 * factor_dim * 0.2)))
        
        self.q_F_scale_mu = self.softplus(nn.Parameter((torch.rand(factor_dim) + 1/2)
                                                        * min(image_dims)
                                                        / (2 * factor_dim)))
        self.q_F_scale_sig = self.softplus(nn.Parameter(self.q_F_scale_mu.data.mean() * 0.05))
        
        
        """
        self.q_c = self.softmax(nn.Parameter(torch.ones(n_data, n_class)))
        self.q_z_0_mu = nn.Parameter(torch.zeros(n_data, z_dim))
        self.q_z_0_sig = self.softplus(nn.Parameter(torch.ones(n_data, z_dim)))
        self.q_z_mu = nn.Parameter(torch.zeros(n_data, T, z_dim))
        self.q_z_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, z_dim)))
        self.q_w_mu = nn.Parameter(torch.zeros(n_data, T, factor_dim))
        self.q_w_sig = self.softplus(nn.Parameter(torch.ones(n_data, T, factor_dim)))
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        self.q_z_F_sig = self.softplus(nn.Parameter(torch.ones(zF_dim)))
        self.q_F_loc_mu = nn.Parameter(torch.zeros(factor_dim, 3))
        self.q_F_loc_sig = self.softplus(nn.Parameter(torch.ones(factor_dim, 3)))
        self.q_F_scale_mu = self.softplus(nn.Parameter(torch.ones(factor_dim)))
        self.q_F_scale_sig = self.softplus(nn.Parameter(torch.ones(1)))
        """
        
        self.use_cuda = use_cuda
        # if on gpu cuda-ize all pytorch (sub)modules
        if use_cuda:
            self.cuda()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        
        # register all PyTorch (sub)modules with pyro
        pyro.module("PDMFA", self)
        
        vox_num = self.voxl_locs.size(0)
        dim_0, dim_1 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,vox_num).view(dim_0, vox_num, dim_1)
        vox_locs = self.voxl_locs.repeat(dim_0, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, 1).repeat(1, vox_num)
        return (-(vox_locs - F_loc_vals).pow(2).sum(dim = 2) / (2 * F_scale_vals.pow(2))).exp()
        
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def model(self, data, u_values, ind, annealing_factor=1.0):
        # data_points = number of data points in mini-batch
        # class_idxs = (data_points, 1)
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + 1, z_dim)
        # w_values = (data_points, time_points, factor_dim)
        # F_loc_values = (data_points, factor_dim, 3)
        # F_scale_values = (data_points, factor_dim)
        # zF_values = (data_points, zF_dim)
        
        
        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("PDMFA", self)
        
        with poutine.scale(scale=annealing_factor):
            z_F = pyro.sample('z_F', Normal(self.p_z_F_mu, self.p_z_F_sig).to_event(1))
        # p(F_mu|z_F) = Normal(F_mu_loc, F_mu_scale)
        # p(F_sig|z_F) = Normal(F_sig_loc, F_sig_scale)
        p_F_loc_mu, p_F_loc_sig, p_F_scale_mu, p_F_scale_sig = self.spat(z_F.view(1,-1))
        p_F_loc_mu = p_F_loc_mu.view(-1, 3)
        p_F_loc_sig = p_F_loc_sig.view(-1, 3)
        p_F_scale_mu = p_F_scale_mu.view(-1)
        p_F_scale_sig = p_F_scale_sig.view(-1)
        
        with pyro.plate("factors", self.factor_dim):
            
            with poutine.scale(scale=annealing_factor):
                F_loc_values = pyro.sample("F_loc_values", Normal(p_F_loc_mu, p_F_loc_sig).to_event(1))
            with poutine.scale(scale=annealing_factor):
                F_scale_values = pyro.sample("F_scale_values", Normal(p_F_scale_mu, p_F_scale_sig))
        
        # f(F): (factor_dim, voxel_num)
        f_F = self.RBF_to_Voxel(F_loc_values, F_scale_values)
           
        
        for i in pyro.plate("data_loop", len(data)):
            
            c_i = pyro.sample("c_{}".format(i), Categorical(probs = self.p_c))

            # p(z_0|c) = Normal(z_0_mu, I)
            z_i_0_mu = self.z_0_mu[c_i]
            z_i_0_sig = self.z_0_sig[c_i]

            with poutine.scale(scale=annealing_factor):
                z_i_0 = pyro.sample("z_{}_0".format(i), Normal(z_i_0_mu, z_i_0_sig).to_event(1))
            
            z_i_t_1 = z_i_0.view(1,-1)
            w_i = torch.Tensor([]).reshape(0, self.factor_dim)
            for j in range(1, self.T + 1):
                
                # p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale)
                z_i_t_mu, z_i_t_sig = self.trans(z_i_t_1, u_values[i,j-1].view(1,-1))
                z_i_t_mu = z_i_t_mu.view(-1)
                z_i_t_sig = z_i_t_sig.view(-1)
                
                with poutine.scale(scale=annealing_factor):
                    z_i_t = pyro.sample("z_{}_{}".format(i,j), Normal(z_i_t_mu, z_i_t_sig).to_event(1))
                
                z_i_t = z_i_t.view(1,-1)
                # p(w_t|z_t) = Normal(w_loc, w_scale)
                w_i_t_mu, w_i_t_sig = self.temp(z_i_t)
                w_i_t_mu = w_i_t_mu.view(-1)
                w_i_t_sig = w_i_t_sig.view(-1)
                
                with poutine.scale(scale=annealing_factor):
                    w_i_t = pyro.sample("w_{}_{}".format(i,j), Normal(w_i_t_mu, w_i_t_sig).to_event(1))
                    
                w_i_t = w_i_t.view(1,-1)
                
                w_i = torch.cat((w_i, w_i_t), dim = 0) 
                z_i_t_1 = z_i_t
        
            # p(y|w,F) = Normal(w*f(F), sigma)
            # w : (time_points, factor_dim)
            # f(F): (factor_dim, voxel_num)
            y_i_mu = torch.matmul(w_i, f_F)
            pyro.sample("y_{}".format(i), Normal(y_i_mu, self.sig_obs).to_event(2), obs = data[i])
        
                
    # the guide q(w_{n,t})q(z_{n,t})p(z_{n,0})q(c_n)q(F)q(z_F) 
    # (i.e. the variational distribution)
    def guide(self, data, u_values, ind, annealing_factor):
        
        # data_points = number of data points in mini-batch
        # mini_batch : (data_points, time_points, voxels)
        # mini_batch_idxs : indices of data points
        
        # register all PyTorch (sub)modules with pyro
        pyro.module("PDMFA", self)
        
        with poutine.scale(scale=annealing_factor):
            pyro.sample('z_F', Normal(self.q_z_F_mu, self.q_z_F_sig).to_event(1))
        
        with pyro.plate("factors", self.factor_dim):
            
            with poutine.scale(scale=annealing_factor):
                pyro.sample("F_loc_values", Normal(self.q_F_loc_mu, self.q_F_loc_sig).to_event(1))
                
            with poutine.scale(scale=annealing_factor):
                pyro.sample("F_scale_values", Normal(self.q_F_scale_mu, self.q_F_scale_sig))
        
        for i in pyro.plate("data_loop", len(data), subsample = ind):
            
            pyro.sample("c_{}".format(i), Categorical(probs = self.q_c[i]))
                
            with poutine.scale(scale=annealing_factor):
                pyro.sample("z_{}_0".format(i), Normal(self.q_z_0_mu[i], self.q_z_0_sig[i]).to_event(1))
            
            for j in pyro.plate("time_loop_{}".format(i), self.T):
                
                with poutine.scale(scale=annealing_factor):
                    pyro.sample("z_{}_{}".format(i,j+1), Normal(self.q_z_mu[i,j], self.q_z_sig[i,j]).to_event(1))
                    
                with poutine.scale(scale=annealing_factor):
                    pyro.sample("w_{}_{}".format(i,j+1), Normal(self.q_w_mu[i,j], self.q_w_sig[i,j]).to_event(1))
                
                
                
"""
PYRO Module
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""              





def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    
    KLD = 1/2 * ( 2 * (p_sigma/q_sigma).log() 
                    - 1
                    + (q_sigma/p_sigma).pow(2)
                    + ( (p_mu - q_mu) / p_sigma ).pow(2) )
    
    return KLD.sum(dim = -1)

def KLD_Cat(q, p):
    
    # sum (q log (q/p) )
    KLD = q * (q / p).log()
    
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
              qs_F_scale_mu, qs_F_scale_sig,
              p_F_scale_mu, p_F_scale_sig,
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
    KL_F_scale = KLD_Gaussian(qs_F_scale_mu, qs_F_scale_sig,
                              p_F_scale_mu, p_F_scale_sig).sum()
    KL_z_F = KLD_Gaussian(qs_z_F_mu, qs_z_F_sig,
                          ps_z_F_mu, ps_z_F_sig)
    beta = annealing_factor
    
    return rec_loss + beta * (KL_c + KL_z_0 + KL_z + KL_w + KL_F_loc + KL_F_scale + KL_z_F)


'Code starts Here'
# we have N number of data points each of size (T*V)
# we have N number of u each of size (T*u_dim)
# we specify a vector with values 0, 1, ..., N-1
# each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)


# setting hyperparametrs

n_data = 200 # number of data points 
T = 10 # number of time points in each sequence
factor_dim = 5 # number of Gaussian blobs (spatial factors)
z_dim = 2 # dimension of temporal latent variable z
emission_dim = 3 # hidden units from z to w (z_dim to factor_dim)
u_dim = 1 # dimension of stimuli embedding
transition_dim = 2 # hidden units from z_{t-1} to z_t
zF_dim = 2 # dimension of spatial factor embedding
n_class = 2 # number of major clusters
sigma_obs = 1e-2 # standard deviation of observation noise
T_A = 30 # annealing iterations <= epoch_num
use_cuda = False # set to True if using gpu
Restore = False # set to True if already trained
batch_size = 50 # batch size
epoch_num = 50 # number of epochs
num_workers = 0 # number of workers to process dataset
lr = 1e-3 # learning rate for adam optimizer

"""
'dummy dataset: comment for real dataset'
v_i, v_j, v_k = 50, 50, 50
V = v_i * v_j * v_k # number of voxels
training_set = [(torch.FloatTensor(torch.randn(T, V)),
                 torch.FloatTensor(torch.randint(0, 2, (T, u_dim))),
                 torch.LongTensor([i])) for i in range(n_data)]
voxel_locations = torch.FloatTensor(
                                    [[i - v_i/2, j - v_j/2, k - v_k/2] 
                                     for i in range(v_i)
                                     for j in range(v_j)
                                     for k in range(v_k)])
'dummy dataset--end--'
"""


"""synthetic dataset-- t prefix means target"""

image_dims = (30,30,30)

classes,\
z_0_values,\
z_values,\
w_values,\
F_loc_mu, F_loc_sig,\
F_scale_mu, F_scale_sig,\
z_0_mu, z_0_sig,\
p_c,\
u_values,\
y,\
training_set,\
voxel_locations = SyntheticNeuralGenerator(n_data = n_data,
                                     T = T,
                                     factor_dim = factor_dim,
                                     z_dim = z_dim,
                                     u_dim = u_dim,
                                     emission_dim = emission_dim,
                                     transition_dim = transition_dim,
                                     n_class = n_class,
                                     sigma_obs = sigma_obs,
                                     image_dims = image_dims, 
                                     ratio_max_space = 0.2,
                                     ratio_sig = 0.05).model()

""" Save one Datapoint for Visulization """

export_PATH = './NIFTI_files/'
if not os.path.exists(export_PATH):
    os.makedirs(export_PATH)
    
NIFTI_image = y[0].reshape(T, image_dims[0], image_dims[1], image_dims[2])
NIFTI_image = NIFTI_image.detach().numpy().transpose(1,2,3,0)
NIFTI_image = nib.Nifti1Image(NIFTI_image, np.eye(4))
NIFTI_image.to_filename(export_PATH + 'y_%d.nii.gz' %(0))

""" end """


# dataset parametrs
'''uncomment for real data
root_dir = './Data/'
list_IDs = [i for i in range(len(glob(root_dir + 'fMRI_*.npz')))]
voxel_locations = torch.FloatTensor(np.load(root_dir + 'vox_locs.npz')['arr_0'])
'''


"""
DMFA SETUP & Training
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""
"""
# Path parameters
save_PATH = './ckpt_files/'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)
    
PATH_DMFA = save_PATH + 'DMFA_epoch%d' %(epoch_num)

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
            image_dims = image_dims,
            voxel_locations = voxel_locations,
            use_cuda = use_cuda)

optim_dmfa = optim.Adam(dmfa.parameters(), lr = lr)

# number of parameters  
total_params = sum(p.numel() for p in dmfa.parameters())
learnable_params = sum(p.numel() for p in dmfa.parameters() if p.requires_grad)
print('Total Number of Parameters: %d' % total_params)
print('Learnable Parameters: %d' %learnable_params)

'''uncomment for real data
training_set = Dataset(list_IDs, root_dir)
'''
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}
train_loader = data.DataLoader(training_set, **params)

if Restore == False:
    
    print("Training...")
    
    for i in range(epoch_num):
        time_start = time.time()
        loss_value = 0.0
        for batch_indx, batch_data in enumerate(train_loader):
        # update DMFA
            
            mini_batch, u_vals, mini_batch_idxs = batch_data
            
            mini_batch = Variable(mini_batch)
            u_vals = Variable(u_vals)
            mini_batch_idxs = Variable(mini_batch_idxs.reshape(-1))
            
            mini_batch = mini_batch.to(device)
            u_vals = u_vals.to(device)
            mini_batch_idxs = mini_batch_idxs.to(device)

            
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
            qs_F_scale_mu, qs_F_scale_sig,\
            p_F_scale_mu, p_F_scale_sig,\
            qs_z_F_mu, qs_z_F_sig,\
            ps_z_F_mu, ps_z_F_sig\
            = dmfa.forward(mini_batch, u_vals, mini_batch_idxs)



        # set gradients to zero in each iteration
            optim_dmfa.zero_grad()
        
        # computing loss
            annealing_factor = 1 #min(1.0, 0.01 + i / T_A) # inverse temperature
            loss_dmfa = ELBO_Loss(mini_batch,
                                  y_hat, 
                                  q_cs, p_cs,
                                  qs_z_0_mus, qs_z_0_sigs,
                                  p_z_0_mu, p_z_0_sig,
                                  q_z_mus, q_z_sigs,
                                  p_z_mu, p_z_sig,
                                  q_w_mus, q_w_sigs,
                                  p_w_mu, p_w_sig,
                                  qs_F_loc_mu, qs_F_loc_sig,
                                  p_F_loc_mu, p_F_loc_sig,
                                  qs_F_scale_mu, qs_F_scale_sig,
                                  p_F_scale_mu, p_F_scale_sig,
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

        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        print('====> Epoch: %d ELBO_Loss : %0.4f'
              % ((i + 1), loss_value / len(train_loader.dataset)))

        torch.save(dmfa.state_dict(), PATH_DMFA)
        
if Restore:
    dmfa.load_state_dict(torch.load(PATH_DMFA)) 
    #, map_location=lambda storage, loc: storage))
 
"""
"""
DMFA SETUP & Training--END
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""


"""
PDMFA SETUP & Training
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""

# setup, training, and evaluation
def main(args):

    data = y
    N_train_data = len(y)
    N_batches = int(N_train_data / batch_size +
                         int(N_train_data % batch_size > 0))

    # instantiate the PDMFA
    pdmfa = PDMFA(n_data = n_data,
                  batch_size = batch_size,
                  T = T,
                  factor_dim = factor_dim,
                  z_dim = z_dim,
                  emission_dim = emission_dim,
                  u_dim = u_dim,
                  transition_dim = transition_dim,
                  zF_dim = zF_dim,
                  n_class = n_class,
                  sigma_obs = sigma_obs,
                  image_dims = image_dims,
                  voxel_locations = voxel_locations,
                  use_cuda = use_cuda)

    # setup optimizer
    adam_params = {"lr": lr, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = Trace_ELBO()
    svi = SVI(pdmfa.model, pdmfa.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        torch.save(pdmfa.state_dict(), args.save_model)
        adam.save(args.save_opt)

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        pdmfa.load_state_dict(torch.load(args.load_model))
        adam.load(args.load_opt)

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_batches + 1) /
                 float(args.annealing_epochs * N_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * batch_size, N_train_data])
        ind = shuffled_indices[mini_batch_start:mini_batch_end]
        # do an actual gradient step
        loss = svi.step(data, u_values, ind, annealing_factor)
        # keep track of the training loss
        return loss


    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(epoch_num):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()

        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        print("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / (n_data * T), epoch_time))

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    
    # Path parameters
    save_PATH = './ckpt_files/'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
    
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=10)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.1)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=5)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='./ckpt_files/PDMFA_opt')
    parser.add_argument('-smod', '--save-model', type=str, default='./ckpt_files/PDMFA_model')
    args = parser.parse_args()

    main(args)

