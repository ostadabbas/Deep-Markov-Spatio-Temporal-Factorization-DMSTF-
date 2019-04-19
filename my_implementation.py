# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:16:32 2019

@author: Amir
"""
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import os
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from glob import glob
plt.subplots_adjust(top = 0.99, bottom=0.1, hspace=0.1, wspace=0.1)
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
        i = torch.LongTensor(index)

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
    def __init__(self, n_data=100, T = 10,  factor_dim=10, z_dim=5, emission_dim=5, u_dim=2,
                 transition_dim=5, zF_dim=5, n_class=5, sigma_obs, voxel_locations, use_cuda=False):
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
        self.p_c = torch.ones(n_class) / n_class
        """
        # define Gauss ianpior p(z_F)
        """
        self.p_z_F_mu = torch.zeros(zF_dim)
        self.p_z_F_sig = torch.ones(zF_dim)
        """
        # define trainable parameters that help define
        # the probability distribution p(z_0|c)
        """
        self.z_0_mu = nn.Parameter(torch.zeros(n_class, z_dim))
        self.z_0_sig = torch.ones(n_class, z_dim)
        """
        # define trainable parameters that help define
        # the probability distributions for inference
        # q(c), q(z_0)...q(z_T), q(w_1)...q(w_T), q(z_F), q(F_loc), q(F_scale)
        """
        self.softmax = nn.Softmax(dim = 1)
        self.softplus = nn.Softplus()
        self.q_c = self.softmax(nn.Parameter(torch.zeros(n_data, n_class)))
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

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all pytorch (sub)modules
        if use_cuda:
            self.cuda()

    def RBF_to_Voxel(self, F_loc_values, F_scale_values):
        
        vox_num = self.vox_locs.size(0)
        dim_0, dim_1, dim_2 = F_loc_values.size()
        F_loc_vals = F_loc_values.repeat(1,1,vox_num).view(dim_0, dim_1, vox_num, dim_2)
        vox_locs = self.voxl_locs.repeat(dim_0, dim_1, 1, 1)
        F_scale_vals = F_scale_values.view(dim_0, dim_1, 1).repeat(1, 1, vox_num)
        return torch.exp(-torch.sum(torch.pow(vox_locs - F_loc_vals, 2), dim = 3) / F_scale_vals)
        
        
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
        z_t_1 = z_values[:,:-1,:].view(N_max * T_max, -1)
        u_t_1 = u_values.view(N_max * T_max, -1)
        p_z_mu, p_z_sig = self.trans(z_t_1, u_t_1)
        p_z_mu = p_z_mu.view(N_max, T_max, -1)
        p_z_sig = p_z_sig.view(N_max, T_max, -1)
            
        # p(w_t|z_t) = Normal(w_loc, w_scale)
        z_t = z_values[:,1:,:].view(N_max * T_max, -1)
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
        obs_noise = Variable(y_hat.data.new(y_hat.size()).normal_())
        y_hat = obs_noise.mul(self.sig_obs).add_(y_hat_nn)
        
        return p_z_0_mu, p_z_0_sig,\
                p_z_mu, p_z_sig,\
                p_w_mu, p_w_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                p_F_scale_mu, p_F_scale_sig,\
                self.p_z_F_mu, self.p_z_F_sig,\
                y_hat,\
                p_cs
                
        # note that (both here and elsewhere) we use poutine.scale to take care
        # of KL annealing.
                
    # the guide q(w_{n,t})q(z_{n,t})p(z_{n,0})q(c_n)q(F)q(z_F) 
    # (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_idxs):
        
        # data_points = number of data points in mini-batch
        # mini_batch : (data_points, time_points, voxels)
        # mini_batch_idxs : indices of data points
        
        # this is the number of data points we need to process in the mini-batch
        N_max = mini_batch.size(0)    
        n_class = self.q_c.size(-1)
        
        q_cs = self.q_c[mini_batch_idxs]
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs]
        q_z_0_sigs = self.q_z_0_sig[mini_batch_idxs]
        z_0_values = self.Reparam(q_z_0_mus, q_z_0_sigs)
        z_0_values = z_0_values.unsqueeze(1)
        q_z_mus = self.q_z_mu[mini_batch_idxs] 
        q_z_sigs = self.q_z_sig[mini_batch_idxs]        
        z_t_values = self.Reparam(q_z_mus, q_z_sigs)
        z_values = torch.cat((z_0_values, z_t_values), dim = 1)
        q_w_mus = self.q_w_mu[mini_batch_idxs] 
        q_w_sigs = self.q_w_sig[mini_batch_idxs]        
        w_values = self.Reparam(self.q_w_mu, self.q_w_sig)
        q_F_loc_mus = self.q_F_loc_mu.repeat(N_max, 1, 1)
        q_F_loc_sigs = self.q_F_loc_sig.repeat(N_max, 1, 1)
        F_loc_values = self.Reparam(q_F_loc_mus, q_F_loc_sigs)
        q_F_scale_mus = self.q_F_scale_mu.repeat(N_max, 1)
        q_F_scale_sigs = self.q_F_scale_sig.repeat(q_F_scale_mus.size())
        F_scale_values = self.Reparam(q_F_scale_mus, q_F_scale_sigs)
        q_z_F_mus = self.q_z_F_mu.repeat(N_max, 1)
        q_z_F_sigs = self.q_z_F_sig.repeat(N_max, 1)
        zF_values = self.Reparam(q_z_F_mus, q_z_F_sigs)
        
        return z_values,\
                w_values,\
                F_loc_values,\
                F_scale_values,\
                zF_values,\
                q_cs,\
                q_z_0_mus.repeat(1, n_class).view(N_max, n_class, -1),\
                q_z_0_sigs.repeat(1, n_class).view(N_max, n_class, -1),\
                q_z_mus, q_z_sigs,\
                self.q_F_loc_mu, self.q_F_loc_sig,\
                self.q_F_scale_mu.view(-1,1),\
                self.q_F_scale_sig.repeat(self.q_F_scale_mu.size()).view(-1,1),\
                self.q_z_F_mu, self.q_z_F_sig
                
                
                

def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    
    KLD = 1/2 * ( 2 * (p_sigma/q_sigma).log() 
                    - q_mu.size(-1)
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
              q_s_z_0_mus, qs_z_0_sigs,
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
              ps_z_F_mu, ps_z_F_sig):
    
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
    
    return rec_loss - KL_c - KL_z_0 - KL_z - KL_w - KL_F_loc - KL_F_scale - KL_z_F

# we have N number of data points each of size (T*V)
# we have N number of u each of size (T*u_dim)
# we specify a vector with values 0, 1, ..., N-1
# each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)
    
Restore = False
#Path parameters
save_PATH = './ckpt_files/'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)
    
PATH_DMFA = save_PATH + '/DMFA_epoch%d' %(epoch_num)

dmfa = DMFA(n_data=100,
            T = 10,
            factor_dim=10,
            z_dim=5,
            emission_dim=5,
            u_dim=2,
            transition_dim=5,
            zF_dim=5,
            n_class=5,
            sigma_obs,
            voxel_locations,
            use_cuda = False)

optim_dmfa = optim.Adam(dmfa.parameters(), lr=1e-3)

training_set = Dataset(list_IDs, root_dir)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}
train_loader = data.DataLoader(training_set, **params)

if Restore == False:
    
    print("Training...")
    
    for i in range(epoch_num):
        time_start = time.time()
        loss_value = 0.0
        for batch_indx, batch_data in enumerate(train_loader):
        # update DMFA
            
            batch_data = Variable(batch_data)
            data_ae = batch_data.to(device)


            z_values,\
            w_values,\
            F_loc_values,\
            F_scale_values,\
            zF_values,\
            q_cs,\
            qs_z_0_mus,\
            qs_z_0_sigs,\
            q_z_mus, q_z_sigs,\
            qs_F_loc_mu, qs_F_loc_sig,\
            qs_F_scale_mu,\
            qs_F_scale_sig,\
            qs_z_F_mu, qs_z_F_sig



            dmfa.guide(mini_batch, mini_batch_idxs)

            optim_mixed.zero_grad()
        
            # get output from both modules	
            weight, encoded_feats, reconstructed_output, encoder_bias, decoder_bias = tied_module_mixed.forward(data_ae)
            
            # back propagation
            AutoEncoder_loss, L2Loss, sparsity_loss, overfit_loss = TiedAutoEncoderLoss(weight, encoded_feats, reconstructed_output, data_ae)
            AutoEncoder_loss.backward()

            optim_mixed.step()

            loss_value += AutoEncoder_loss.data[0] 
            L2Loss_value += L2Loss.item()
            sparsity_loss_value += sparsity_loss.item()
            overfit_loss_value += overfit_loss.item()

    time_end = time.time()
    print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
    print('====> Epoch: %d Obj_Loss : %0.8f | L2_Loss : %0.8f | Sparsity_Loss : %0.8f | Overfit_Loss : %0.8f'\
          % ((i + 1), loss_value / len(train_loader.dataset),\
             L2Loss_value / len(train_loader.dataset),\
             sparsity_loss_value / len(train_loader.dataset),\
             overfit_loss_value / len(train_loader.dataset)))

    torch.save(tied_module_mixed.state_dict(), PATH_AutoEnc)