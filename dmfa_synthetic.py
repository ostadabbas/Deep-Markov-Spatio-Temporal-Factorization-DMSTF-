import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import os
import time
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))
from generate_fMRI_patches import *
import nibabel as nib
from matplotlib.patches import Ellipse
    


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
        y = np.load(os.path.join(self.root_dir, 'fMRI_%.5d.npz' %ID))['arr_0']
        s = np.zeros((T,u_dim))

        y = torch.FloatTensor(y)
        s = torch.FloatTensor(s)
        i = torch.LongTensor([index])

        return (y,s,i)

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
        self.relu = nn.ReLU()
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
        self.relu = nn.ReLU()
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
        self.relu = nn.ReLU()
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
                 maxima_locs = None,
                 voxel_locations = None, use_cuda=False):
        super(DMFA, self).__init__()
        
        self.im_dims = image_dims
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
        self.p_c = torch.ones(n_class)
        """
        # define Gaussian pior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim)
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        
        self.z_0_mu = nn.Parameter((torch.rand(n_class, z_dim) - 1/2) * 1)
        
        self.z_0_sig = ((torch.ones(n_class, z_dim) / (2 * n_class) * 0.1 * 5).exp() - 1).log()
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
        self.q_z_0_sig = init_sig
        self.q_z_mu = nn.Parameter(torch.rand(n_data, T, z_dim) - 1/2)
        init_sig = ((torch.ones(n_data, T, z_dim) / (2 * n_class) * 0.1).exp() - 1).log()
        self.q_z_sig = init_sig
        self.q_w_mu = nn.Parameter(torch.rand(n_data, T, factor_dim)- 1/2)
        init_sig = ((torch.ones(n_data, T, factor_dim) / (2 * n_class) * 0.1).exp() - 1).log()
        self.q_w_sig = init_sig
        self.q_z_F_mu = nn.Parameter(torch.zeros(zF_dim))
        init_sig = (torch.ones(zF_dim).exp() - 1).log()
        self.q_z_F_sig = nn.Parameter(init_sig)
        if maxima_locs is not None:
            self.q_F_loc_mu = nn.Parameter(torch.FloatTensor(maxima_locs[::len(maxima_locs)//factor_dim][:factor_dim]))
        else:
            self.q_F_loc_mu = nn.Parameter((torch.rand(factor_dim, 3) - 1/2) 
                                            * torch.FloatTensor(image_dims))
        init_sig = ((torch.ones(factor_dim, 3) * torch.FloatTensor(image_dims) / (2 * factor_dim) * 0.2).exp() - 1).log()
        self.q_F_loc_sig = init_sig
        
        init_sig = (((torch.rand(factor_dim) + 1/2) * min(image_dims) / (2 * 2.5)).exp() - 1).log()
        self.q_F_scale_mu = nn.Parameter(init_sig)
        init_sig = ((self.softplus(self.q_F_scale_mu).data.mean() * 0.05).exp() - 1).log()
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
        N_max = u_values.size(0)
        T_max = u_values.size(1)
        
        # p(c) = Uniform(n_class)
        p_cs = self.smax(self.p_c).repeat(N_max, 1)
        
        # p(z_0|c) = Normal(z_0_mu, I)
        p_z_0_mu = self.z_0_mu.repeat(N_max, 1, 1)
        p_z_0_sig = self.softplus(self.z_0_sig).repeat(N_max, 1, 1)
        
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
        
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs]
        q_z_0_sigs = self.softplus(self.q_z_0_sig[mini_batch_idxs])
        q_z_mus = self.q_z_mu[mini_batch_idxs]
        q_z_sigs = self.softplus(self.q_z_sig[mini_batch_idxs])        
        q_w_mus = self.q_w_mu[mini_batch_idxs] 
        q_w_sigs = self.softplus(self.q_w_sig[mini_batch_idxs])
        self.q_F_loc_mu.data = torch.clamp(self.q_F_loc_mu, min = -min(self.im_dims)/2, max = min(self.im_dims)/2)        
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
        q_cs = self.softmax(q_cs_Unlog)
        
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
    
    return rec_loss + beta * (KL_c + KL_z_0 + KL_z + KL_w + KL_F_loc + KL_F_scale + KL_z_F)


'Code starts Here'
# we have N number of data points each of size (T*V)
# we have N number of u each of size (T*u_dim)
# we specify a vector with values 0, 1, ..., N-1
# each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)


# setting hyperparametrs

T = 5 # number of time points in each sequence
factor_dim = 30 # number of Gaussian blobs (spatial factors)
z_dim = 2 # dimension of temporal latent variable z
emission_dim = 8 # hidden units from z to w (z_dim to factor_dim)
u_dim = 1 # dimension of stimuli embedding
transition_dim = 2 # hidden units from z_{t-1} to z_t
zF_dim = 2 # dimension of spatial factor embedding
n_class = 3 # number of major clusters
sigma_obs = 1e-6 # standard deviation of observation noise
T_A = 20 # annealing iterations <= epoch_num
use_cuda = False # set to True if using gpu
Restore = True # set to True if already trained
batch_size = 10 # batch size
epoch_num = 300 # number of epochs
num_workers = 0 # number of workers to process dataset
lr = 1e-2 # learning rate for adam optimizer

# dataset parametrs
root_dir = './data_synthetic/'
list_IDs = [i for i in range(30)]
voxel_locations = torch.FloatTensor(np.load(root_dir + 'vox_locs.npz')['arr_0'])
maxima_locs = torch.FloatTensor(np.load(root_dir + 'maxima_locs.npz')['arr_0'])
image_dims = torch.FloatTensor(np.load(root_dir + 'image_dims.npz')['arr_0'])

training_set = Dataset(list_IDs, root_dir)
n_data = len(training_set)
sigma_obs_model = sigma_obs * 10
classes = torch.LongTensor([0,1,2]*(n_data//3))

"""
DMFA SETUP & Training
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""
# Path parameters
save_PATH = './ckpt_fMRI/'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)
    
PATH_DMFA = save_PATH + 'DMFA_synthetic'

dmfa = DMFA(n_data = n_data,
            T = T,
            factor_dim = factor_dim,
            z_dim = z_dim,
            emission_dim = emission_dim,
            u_dim = u_dim,
            transition_dim = transition_dim,
            zF_dim = zF_dim,
            n_class = n_class,
            sigma_obs = sigma_obs_model,
            image_dims = image_dims,
            voxel_locations = voxel_locations,
            use_cuda = use_cuda,
            maxima_locs = maxima_locs)

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
        if True and i%10==0:
            fig_PATH = './synthetic_results/'
            if not os.path.exists(fig_PATH):
                os.makedirs(fig_PATH)
            
            fig = plt.figure()
            z_0 = dmfa.q_z_0_mu.detach().numpy()
            z_0_p = dmfa.z_0_mu.detach().numpy()
            z_0_p_sig = nn.Softplus()(dmfa.z_0_sig).detach().numpy()
            colors = ['g','b','r','y']
            labels = ['group1','group2','group3']
            c_idx = classes.detach().numpy()
            ax = fig.add_subplot(111)
            ax.set_title("z_0");
            for j in range(n_class):
                ax.scatter(z_0[c_idx==j,0],z_0[c_idx==j,1], label = labels[j])
                circle = Ellipse((z_0_p[j, 0], z_0_p[j, 1]), z_0_p_sig[j,0], z_0_p_sig[j,1], color=colors[j], alpha = 0.2)
                ax.add_artist(circle)
            ax.legend()
            fig.savefig(fig_PATH + "q_z_0{%.3d}.png" %i)
            plt.close("all")
        
if Restore:
    dmfa.load_state_dict(torch.load(PATH_DMFA
    , map_location=lambda storage, loc: storage))
    
    params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}
    train_loader = data.DataLoader(training_set, **params)
    
    batch_indx, batch_data = next(iter(enumerate(train_loader)))
        
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
    
    print("class Assignments are:")
    print(torch.argmax(dmfa.q_c, dim = 1))
    
    fig_PATH = './synthetic_results/'
    if not os.path.exists(fig_PATH):
        os.makedirs(fig_PATH)
    
    fig = plt.figure()
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    z_0_p = dmfa.z_0_mu.detach().numpy()
    z_0_p_sig = nn.Softplus()(dmfa.z_0_sig).detach().numpy()*2
    colors = ['r','b','g','y']
    labels = ['group1','group2','group3']
    c_idx = classes.detach().numpy()
    ax = fig.add_subplot(111)
    ax.set_title("$z^w_0$", fontsize=16);
    for j in range(n_class):
        ax.scatter(z_0[c_idx==j,0],z_0[c_idx==j,1], label = labels[j])
        circle = Ellipse((z_0_p[j, 0], z_0_p[j, 1]), z_0_p_sig[j,0], z_0_p_sig[j,1], color=colors[j], alpha = 0.2)
        ax.add_artist(circle)
    ax.legend(framealpha = 0, fontsize=12)
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            right='off', left='off', labelleft='off') # labels along the bottom edge are off
    fig.savefig(fig_PATH + "q_z_0.pdf")
    zs = dmfa.q_z_mu.detach().numpy()
    z_0_p = dmfa.z_0_mu.detach().numpy()
    z_0s = np.expand_dims(z_0_p,1)
    for j in range(1, T+1, 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("$z^w_%d$" %j, fontsize=16);
        z_0_p, z_0_p_sig = dmfa.trans(torch.Tensor(z_0_p), torch.zeros(n_class,u_dim))
        z_0_p = z_0_p.detach().numpy()
        z_0_p_sig = z_0_p_sig.detach().numpy()*4
        z_0s = np.concatenate((z_0s, np.expand_dims(z_0_p, 1)), axis = 1)
        for k in range(n_class):
            ax.scatter(zs[c_idx==k,j-1,0],zs[c_idx==k,j-1,1], label = labels[k])
            circle = Ellipse((z_0_p[k, 0], z_0_p[k, 1]), z_0_p_sig[k,0], z_0_p_sig[k,1], color=colors[k], alpha = 0.2)
            ax.add_artist(circle)
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        right='off', left='off', labelleft='off') # labels along the bottom edge are off
        ax.legend(framealpha = 0,fontsize=12)
        fig.savefig(fig_PATH + "q_z_%d.pdf" %j)
    
    zss = np.concatenate((np.expand_dims(z_0, 1), zs), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Latent Space Trajectory");
    labels = [['group1/1','group1/2'],['group2/1','group2/2'],['group3/1','group3/2']]
    for k in range(n_class):
        k_idxs = np.where(c_idx==k)[0]
        k_idxs = k_idxs[::5]
        colorss = np.zeros((len(k_idxs), 3))
        c_id = [2,0,1]
        colorss[:,c_id[k]] = np.arange(0.5, 1, 0.5/len(k_idxs))[:len(k_idxs)]
        for jj, j in enumerate(k_idxs):
            ax.scatter(zss[j,:,0],zss[j,:,1], s=np.array([1,10,20,30,43,56])*2, color=colorss[jj], label = labels[k][jj])
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        right='off', left='off', labelleft='off') # labels along the bottom edge are off
    ax.legend(framealpha = 0)
    fig.savefig(fig_PATH + "trajectory.pdf")
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Mean Latent Space Trajectory", fontsize=12);
    labels = ['group1','group2','group3']
    colors = ['r','b','g','y']
    for k in range(n_class):
        ax.quiver(z_0s[k,:-1,0],z_0s[k,:-1,1],
                  z_0s[k,1:,0]-z_0s[k,:-1,0], z_0s[k,1:,1]-z_0s[k,:-1,1],
                  color=colors[k], label = labels[k],
                  scale_units='xy', angles='xy', scale=1.05)
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        right='off', left='off', labelleft='off') # labels along the bottom edge are off
    ax.legend(framealpha = 0)
    fig.savefig(fig_PATH + "trajectory_new.pdf")

    print("Blob centers are:")
    print(dmfa.q_F_loc_mu + torch.Tensor([i/2 for i in image_dims]))
    
    print("Blob width are:")
    print(dmfa.q_F_scale_mu)
    
    plt.figure()
    plt.title("real vs recon fMRI for time point 0")
    plt.plot(mini_batch[0,0,:].detach().numpy())
    plt.plot(y_hat[0,0,:].detach().numpy())
    
    from scipy.io import loadmat
    XC = loadmat('./data_synthetic/X.mat')['X']
    CX = loadmat('./data_synthetic/C.mat')['C']
    ws = dmfa.q_w_mu.reshape(-1,30).detach().numpy()
    idx_C, idx_w = np.unravel_index(np.corrcoef(CX.transpose(),
                                                ws.transpose())[:30,30:].flatten().argsort()[-1], (30,30))
    ratio = CX[:,idx_C].std()/ws[:,idx_w].std()
    bias = CX[:,idx_C].mean() - ws[:,idx_w].mean()*ratio
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Sample Dynamics")
    ax.plot(XC[:,idx_C], label = "Nueral Activity")
    ax.plot(CX[:,idx_C], label = "BOLD Signal")
    ax.plot(ws[:,idx_w]*ratio+bias, label = "Recovered Weight")
    ax.legend()

    fig.savefig(fig_PATH + "dynamics.pdf")
    
    
    ws = dmfa.q_w_mu.reshape(-1,30).detach().numpy()
    z_t_1 = dmfa.q_z_mu[-1,-1].reshape(1, -1)
    for i in range(T*10):
        u_t_1 = torch.zeros(1, u_dim)
        p_z_mu, p_z_sig = dmfa.trans(z_t_1, u_t_1)
        p_w_mu, p_w_sig = dmfa.temp(p_z_mu)
        z_t_1 = p_z_mu
        ws = np.concatenate((ws, p_w_mu.detach().numpy()), axis = 0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Sample Prediction")
    ax.plot(XC[:,idx_C], label = "Neural Activity")
    ax.plot(CX[:,idx_C], label = "BOLD Signal")
    ax.plot(ws[:,idx_w]*ratio+bias, label = "Recovered Weight")
    ax.legend()
    fig.savefig(fig_PATH + "prediction.pdf")
    
    
"""
DMFA SETUP & Training--END
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""