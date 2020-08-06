
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import os
import time
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))
import pdb
    

class SyntheticGenerator:
    def __init__(self, 
                 n_data=100,
                 T = 15,
                 factor_dim=2,
                 z_dim=2,
                 u_dim=0,
                 n_class=1,
                 sigma_obs = 1e-5,
                 image_dims = (10,10,10),
                 ratio_sig = 0.0):
        super(SyntheticGenerator, self).__init__()
        
        """ observation noise """
        self.sig_obs = sigma_obs
        self.ratio_sig = ratio_sig
        self.n_data = n_data
        self.T = T
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.n_class = n_class
        """ 3D coordinates of voxels: # of voxels times 3"""
        self.im_dims = image_dims
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
        p_c = torch.ones(n_class) 
        self.p_c = p_c / p_c.sum()
        """
        # p(z_0|c)
        """
        z_0_mu = [[0.0,0.0]]
        z_0_sig = [[1.0,1.0]]
        
        self.z_0_mu = torch.Tensor(np.asarray(z_0_mu))
        self.z_0_sig = torch.Tensor(np.asarray(z_0_sig))
        
        """
        # parameters for generating
        # p(F_loc), p(F_scale)
        """
        F_loc_mu = [[.25,.25,.25],[-.25,-.25,-.25]]
        F_loc_sig = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
        F_scale_mu = [[.1],[0.15]]
            
        self.F_loc_mu = torch.Tensor(np.asarray(F_loc_mu)) * torch.FloatTensor(image_dims) 
        self.F_loc_sig = torch.Tensor(np.asarray(F_loc_sig)) * torch.FloatTensor(image_dims)
        self.F_scale_mu = torch.Tensor(np.asarray(F_scale_mu).reshape(-1)) * min(image_dims)
            
        self.F_scale_sig = (self.F_scale_mu * ratio_sig).mean()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        vox_num = self.voxl_locs.size(0)
        dim_0, dim_1, dim_2 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,1,vox_num).view(dim_0, dim_1, vox_num, dim_2)
        vox_locs = self.voxl_locs.repeat(dim_0, dim_1, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, dim_1, 1).repeat(1, 1, vox_num)
        return torch.exp(-torch.sum(torch.pow(vox_locs - F_loc_vals, 2), dim = 3) / (2 * F_scale_vals.pow(2)))
        
        
    def Reparam(self, mu_latent, sigma_latent):
        eps = mu_latent.data.new(mu_latent.size()).normal_()
        return eps.mul(sigma_latent).add_(mu_latent)
    
    def model(self, plot_flag = True):
        """ sample class numbers from p_c """
        classes = Categorical(self.p_c).sample((self.n_data,))
        
        """ sample p(z_0|c) """
        z_0_mus = self.z_0_mu[classes]
        z_0_sigs = self.z_0_sig[classes]
        z_0_values = self.Reparam(z_0_mus, z_0_sigs)        

        z_values = torch.zeros(self.n_data, self.T, self.z_dim)
        w_values = torch.zeros(self.n_data, self.T, self.factor_dim)
        z_prev = z_0_values
        for i in range(self.T):
            t_1 = torch.FloatTensor([[0.2,0],[0,0.2]])
            t_2 = torch.FloatTensor([[0,0.5],[0,0]])
            t_3 = torch.FloatTensor([[0,0],[-0.1,0]])
            z_mus = torch.matmul(z_prev, t_1) +\
                    torch.matmul(z_prev, t_2).tanh() +\
                    torch.matmul(z_prev, t_3).sin()
            z_sigs = torch.ones(self.n_data, self.z_dim)
            z_prev = self.Reparam(z_mus, z_sigs)
            z_values[:,i,:] = z_prev
            w_mus = 0.5 * z_prev
            w_sigs = 0.1 * torch.ones(self.n_data, self.z_dim)
            w_values[:,i,:] = self.Reparam(w_mus, w_sigs)
            
        F_loc_mus = self.F_loc_mu.repeat(self.n_data, 1, 1)
        F_loc_sigs = self.F_loc_sig.repeat(self.n_data, 1, 1)
        F_loc_values = self.Reparam(F_loc_mus, F_loc_sigs)
        F_loc_values = torch.clamp(F_loc_values, min = -min(self.im_dims)/2, max = min(self.im_dims)/2)
        F_scale_mus = self.F_scale_mu.repeat(self.n_data, 1)
        F_scale_sigs = self.F_scale_sig.repeat(self.n_data, self.factor_dim)
        F_scale_values = self.Reparam(F_scale_mus, F_scale_sigs)
        F_scale_values = torch.clamp(F_scale_values, min = 1e-1)

        f_F = self.RBF_to_Voxel(F_loc_values, F_scale_values)
        y_nn = torch.matmul(w_values, f_F)
        obs_noise = y_nn.data.new(y_nn.size()).normal_()
        y = obs_noise.mul(self.sig_obs * y_nn.abs()).add_(y_nn)
        
        training_set = [(y[i], self.u_values[i], torch.LongTensor([i]))
                        for i in range(self.n_data)]
        return classes,\
                z_0_values,\
                z_values,\
                w_values,\
                y,\
                training_set,\
                self.voxl_locs

class TemporalFactors(nn.Module):
    """
    Parameterizes the Gaussian weight p(w_t | z_t)
    """
    def __init__(self):
        super(TemporalFactors, self).__init__()
        self.param_loc = 0.5
        self.param_scale = 0.1
    def forward(self, z_t):
        """
        Given the latent z_t corresponding to the time step t
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(w_t | z_t)
        """
        # compute the 'weight mean' given z_t
        weight_loc = self.param_loc * z_t
        
        # compute the weight scale using the weight loc
        # from above as input. The softplus ensures that scale is positive
        weight_scale = self.param_scale * torch.ones(weight_loc.size())
        # return loc, scale of w_t which can be fed into Normal
        return weight_loc, weight_scale
    
    
class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """
    def __init__(self):
        super(GatedTransition, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.lin = nn.Parameter(torch.zeros(1))
        self.param_scale = 1.0
    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} and stimuli u_{t-1} corresponding to the time
        step t-1, we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1}, u_{t-1})
        """
        t_1 = torch.FloatTensor([[0,0],[0,0]])
        t_1[0,0] = self.lin
        t_1[1,1] = self.lin
        t_2 = torch.FloatTensor([[0,0],[0,0]])
        t_2[0,1] = self.alpha
        t_3 = torch.FloatTensor([[0,0],[0,0]])
        t_3[1,0] = self.beta
        z_loc = torch.matmul(z_t_1, t_1) +\
                torch.matmul(z_t_1, t_2).tanh() +\
                torch.matmul(z_t_1, t_3).sin()
        z_scale = self.param_scale * torch.ones(z_loc.size())
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
                 u_dim=2,
                 zF_dim=5, n_class=5, sigma_obs = 1e-2, image_dims = None,
                 maxima_locs = None,
                 voxel_locations = None, use_cuda=False):
        super(DMFA, self).__init__()
        
        self.im_dims = image_dims
        # observation noise
        self.sig_obs = sigma_obs
        # 3D coordinates of voxels: # of voxels times 3
        self.voxl_locs = voxel_locations
        # instantiate pytorch modules used in the model and guide below
        self.temp = TemporalFactors()
        self.trans = GatedTransition()
        self.spat = SpatialFactors(factor_dim, zF_dim)
        """
        # define uniform pior p(c)
        """
        self.p_c = torch.ones(n_class) / n_class
        """
        # define Gaussian pior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim)
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        
        self.z_0_mu = torch.zeros(n_class, z_dim)
        self.z_0_sig = torch.ones(n_class, z_dim)
        """
        # define trainable parameters that help define
        # the probability distributions for inference
        # q(c), q(z_0)...q(z_T), q(w_1)...q(w_T), q(z_F), q(F_loc), q(F_scale)
        """
        self.softmax = nn.Softmax(dim = 1)
        self.softplus = nn.Softplus()
        
        self.q_c = torch.ones(n_data, n_class) / n_class
        self.q_z_0_mu = nn.Parameter(torch.rand(n_data, z_dim)- 1/2)
        init_sig = ((torch.ones(n_data, z_dim) / (2 * n_class) * 0.1).exp() - 1).log()
        self.q_z_0_sig = nn.Parameter(init_sig)
        self.q_z_mu = nn.Parameter(torch.rand(n_data, T, z_dim) - 1/2)
        init_sig = ((torch.ones(n_data, T, z_dim) / (2 * n_class) * 0.1).exp() - 1).log()
        self.q_z_sig = nn.Parameter(init_sig)
        self.q_w_mu = nn.Parameter(torch.rand(n_data, T, factor_dim)- 1/2)
        init_sig = ((torch.ones(n_data, T, factor_dim) / (2 * n_class) * 0.1).exp() - 1).log()
        self.q_w_sig = nn.Parameter(init_sig)
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        init_sig = (torch.ones(zF_dim).exp() - 1).log()
        self.q_z_F_sig = nn.Parameter(init_sig)
        if maxima_locs is not None:
            self.q_F_loc_mu = nn.Parameter(torch.FloatTensor(maxima_locs[::len(maxima_locs)//factor_dim][:factor_dim]))
        else:
            self.q_F_loc_mu = nn.Parameter((torch.rand(factor_dim, 3) - 1/2) 
                                            * torch.FloatTensor(image_dims))
            self.q_F_loc_mu.data = torch.FloatTensor([[5,5,5],[-5,-5,-5]])
        init_sig = ((torch.ones(factor_dim, 3) * torch.FloatTensor(image_dims) / (2 * factor_dim) * 0.02).exp() - 1).log()
        self.q_F_loc_sig = init_sig
        
        init_sig = ((torch.FloatTensor([2, 5.5])).exp() - 1).log()
        self.q_F_scale_mu = nn.Parameter(init_sig)
        init_sig = ((self.softplus(self.q_F_scale_mu).data.mean() * 0.05).exp() - 1).log() #Edited
        self.q_F_scale_sig = init_sig
        
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
        eps = Variable(mu_latent.data.new(mu_latent.size()).normal_())
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
        N_max = w_values.size(0)
        T_max = w_values.size(1)
        
        # p(c) = Uniform(n_class)
        p_cs = self.p_c.repeat(N_max, 1)
        
        # p(z_0|c) = Normal(z_0_mu, I)
        p_z_0_mu = self.z_0_mu.repeat(N_max, 1, 1)
        p_z_0_sig = self.z_0_sig.repeat(N_max, 1, 1)
        
        # p(z_t|z_{t-1},u{t-1}) = Normal(z_loc, z_scale)
        z_t_1 = z_values[:,:-1,:].reshape(N_max * T_max, -1)
        p_z_mu, p_z_sig = self.trans(z_t_1)
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
        
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs]
        q_z_0_sigs = self.softplus(self.q_z_0_sig[mini_batch_idxs])
        q_z_mus = self.q_z_mu[mini_batch_idxs]
        q_z_sigs = self.softplus(self.q_z_sig[mini_batch_idxs])        
        q_w_mus = self.q_w_mu[mini_batch_idxs] 
        q_w_sigs = self.softplus(self.q_w_sig[mini_batch_idxs])      
        q_F_loc_mus = self.q_F_loc_mu.repeat(N_max, 1, 1)
        q_F_loc_sigs = self.softplus(self.q_F_loc_sig).repeat(N_max, 1, 1)
        q_F_scale_mus = self.softplus(self.q_F_scale_mu).repeat(N_max, 1)
        q_F_scale_sigs = self.softplus(self.q_F_scale_sig).repeat(q_F_scale_mus.size())
        q_z_F_mus = self.q_z_F_mu.repeat(N_max, 1)
        q_z_F_sigs = self.softplus(self.q_z_F_sig).repeat(N_max, 1)
        
        return q_z_0_mus,\
                q_z_0_sigs,\
                q_z_mus, q_z_sigs,\
                q_w_mus, q_w_sigs,\
                q_F_loc_mus, q_F_loc_sigs,\
                q_F_scale_mus,\
                q_F_scale_sigs,\
                q_z_F_mus, q_z_F_sigs
                
    def forward(self, mini_batch, u_values, mini_batch_idxs):
        
        # get outputs from both modules: guide and model
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
        F_loc_values = torch.clamp(F_loc_values, min = -min(self.im_dims)/2, max = min(self.im_dims)/2)    
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
        n_class = self.p_c.size(-1)
        
        # this is the number of data points we need to process in the mini-batch
        N_max = mini_batch.size(0)
        
        """compute q_c from equation 16 in https://arxiv.org/pdf/1611.05148.pdf"""
        
        z_0_vals = z_0_values.squeeze(1).repeat(1, n_class).view(N_max, n_class, -1)
        
        q_cs_Unlog = p_cs.log() \
                     - 1 / 2 * ((z_0_vals - p_z_0_mu) / p_z_0_sig).pow(2).sum(dim = -1) \
                    - (p_z_0_sig).log().sum(dim = -1)
        q_cs_log = q_cs_Unlog - q_cs_Unlog.logsumexp(dim = -1).view(-1,1).repeat(1, n_class)
        
        q_cs = q_cs_log.exp()
        
        """End"""
        
        self.q_c[mini_batch_idxs,:] = q_cs
        qs_z_0_mus = q_z_0_mus.repeat(1, n_class).view(N_max, n_class, -1)
        qs_z_0_sigs = q_z_0_sigs.repeat(1, n_class).view(N_max, n_class, -1)
        qs_F_loc_mu = self.q_F_loc_mu
        qs_F_loc_sig = self.softplus(self.q_F_loc_sig)
        qs_F_scale_mu = self.softplus(self.q_F_scale_mu).view(-1,1)
        qs_F_scale_sig = self.softplus(self.q_F_scale_sig).repeat(self.q_F_scale_mu.size()).view(-1,1)
        qs_z_F_mu = self.q_z_F_mu
        qs_z_F_sig = self.softplus(self.q_z_F_sig)
        
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

def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    
    KLD = 1/2 * ( 2 * ((p_sigma+1e-4)/(q_sigma+1e-4)).log() 
                    - 1
                    + ((q_sigma+1e-3)/(p_sigma+1e-3)).pow(2)
                    + ( (p_mu - q_mu+1e-3) / (p_sigma+1e-3) ).pow(2) )
    
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
    
    return rec_loss + beta * (KL_c + KL_z_0 + KL_z + KL_w) #+ KL_F_loc + KL_F_scale + KL_z_F)


'Code starts Here'
# we have N number of data points each of size (T*V)
# we have N number of u each of size (T*u_dim)
# we specify a vector with values 0, 1, ..., N-1
# each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)


# setting hyperparametrs

n_data = 100 # number of data points 
T = 15 # number of time points in each sequence
factor_dim = 2 # number of Gaussian blobs (spatial factors)
z_dim = 2 # dimension of temporal latent variable z
u_dim = 0 # dimension of stimuli embedding
zF_dim = 2 # dimension of spatial factor embedding
n_class = 1 # number of major clusters
sigma_obs = 0 #1e-5 # standard deviation of observation noise
T_A = 20 # annealing iterations <= epoch_num
use_cuda = False # set to True if using gpu
Restore = False # set to True if already trained
batch_size = 50 # batch size
epoch_num = 600 # number of epochs
num_workers = 0 # number of workers to process dataset
lr = 1e-2 # learning rate for adam optimizer

# dataset parametrs
##uncomment for real data
image_dims = (30,30,30)

classes,\
z_0_values,\
z_values,\
w_values,\
y,\
training_set,\
voxel_locations = SyntheticGenerator(n_data = n_data,
                                     T = T,
                                     factor_dim = factor_dim,
                                     z_dim = z_dim,
                                     u_dim = u_dim,
                                     n_class = n_class,
                                     sigma_obs = sigma_obs,
                                     image_dims = image_dims, 
                                     ratio_sig = 0.0
                                     ).model()

sigma_obs_model = sigma_obs * (y.abs().mean() + y.abs().std())

"""
DMFA SETUP & Training
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""
# Path parameters
save_PATH = './ckpt_toy/'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)
    
PATH_DMFA = save_PATH + 'DMFA_epoch%d' %(epoch_num)

dmfa = DMFA(n_data = n_data,
            T = T,
            factor_dim = factor_dim,
            z_dim = z_dim,
            u_dim = u_dim,
            zF_dim = zF_dim,
            n_class = n_class,
            sigma_obs = sigma_obs_model,
            image_dims = image_dims,
            voxel_locations = voxel_locations,
            use_cuda = use_cuda)

optim_dmfa = optim.Adam(dmfa.parameters(), lr = lr)

# number of parameters  
total_params = sum(p.numel() for p in dmfa.parameters())
learnable_params = sum(p.numel() for p in dmfa.parameters() if p.requires_grad)
print('Total Number of Parameters: %d' % total_params)
print('Learnable Parameters: %d' %learnable_params)

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}
train_loader = data.DataLoader(training_set, **params)

if Restore == False:
    fig_PATH = './toy_results/'
    if not os.path.exists(fig_PATH):
        os.makedirs(fig_PATH)
    alphas = np.array([None for _ in range(epoch_num)])
    betas = np.array([None for _ in range(epoch_num)])
    lins = np.array([None for _ in range(epoch_num)])
    facs_mu = np.array([None for _ in range(epoch_num)])
    facs_sig = np.array([None for _ in range(epoch_num)])
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
            annealing_factor = min(1.0, 0.01 + i / T_A) # inverse temperature
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
        
        acc = torch.sum(dmfa.q_c.argmax(dim=1)==classes).float()/n_data
        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        print('====> Epoch: %d ELBO_Loss : %0.4f Acc: %0.2f'
              % ((i + 1), loss_value / len(train_loader.dataset), acc))
        
        torch.save(dmfa.state_dict(), PATH_DMFA)
        alphas[i] = dmfa.trans.alpha.item()
        betas[i] = dmfa.trans.beta.item()
        lins[i] = dmfa.trans.lin.item()
        facs_mu[i] = (dmfa.q_F_loc_mu - torch.FloatTensor([[7.5,7.5,7.5],[-7.5,-7.5,-7.5]])).pow(2).mean().sqrt().item() 
        facs_sig[i] = (nn.Softplus()(dmfa.q_F_scale_mu) - torch.FloatTensor([3,4.5])).pow(2).mean().sqrt().item() 
        
        if i % 50 == 0:
            lw = 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('parameter estimation', fontsize=16)
            ax.set_xlabel("epoch number", fontsize=16) 
            ax.tick_params(axis="y", labelcolor="r")
            ax.plot(facs_mu[:500], "r-^", label='factors location RMSE', markevery = 50, linewidth = lw, markersize = 5)
            ax.plot(facs_sig[:500], "r->", label='factors scale RMSE', markevery = 50, linewidth = lw, markersize = 5)
            ax.legend(framealpha = 0, loc='upper left')
            ax.set_ylim(0,3)
            ax.margins(x=0.03)
            
            ax2 = ax.twinx()
            ax2.tick_params(axis="y", labelcolor="b")
            ax2.plot(alphas[:500], "b-^", label=r'$\alpha$', markevery = 50, linewidth = lw, markersize = 5)
            ax2.plot(np.arange(150,500,1),[0.5]*350, "b--",alpha = 0.5, linewidth = 0.5)
            ax2.text(400,0.42,r'$\alpha^* = 0.5$', fontsize=12)
            ax2.plot(betas[:500], "b->", label=r'$\beta$', markevery = 50, linewidth = lw, markersize = 5)
            ax2.plot(np.arange(150,500,1),[-.1]*350, "b--",alpha = 0.5, linewidth = 0.5)
            ax2.text(400,-.05,r'$\beta^* = -0.1$', fontsize=12)
            ax2.plot(lins[:500], "b-<", label=r'$\rho$', markevery = 50, linewidth = lw, markersize = 5)
            ax2.plot(np.arange(150,500,1),[0.2]*350, "b--", alpha = 0.5, linewidth = 0.5)
            ax2.text(400,0.23,r'$\rho^* = 0.2$', fontsize=12)
            ax2.legend(framealpha = 0, loc='upper right')
            ax2.set_ylim(-0.3,1)  
            ax2.margins(x=0.03)
            
            fig.savefig(fig_PATH + "toy_errors_%d.pdf" %i)
            plt.close('all')
         
    
"""
DMFA SETUP & Training--END
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""