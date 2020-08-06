

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import nibabel as nib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
import os

def plot_result_depression_14(dmfa, classes, root_dir, list_IDs,
                              fig_PATH, mini_batch, y_hat,
                              NIFTI_PATH):
    n_class = dmfa.p_c.size(-1)
    T = dmfa.q_w_mu.size(1)
    voxel_locations = dmfa.voxl_locs
    image_dims = dmfa.im_dims
    print("class Assignments are:")
    print(torch.argmax(dmfa.q_c, dim = 1))
    
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    fig = plt.figure()
    c_idx = classes.detach().numpy()
    # to read json file:
    with open(root_dir + 'meta_data.json', 'r') as f:
        meta_data = json.load(f)
    m_idx = (np.array(meta_data['music_type'])[list_IDs]+1)/2    
    
    colorss = [['cyan','steelblue'],['salmon','orangered']]
    labelss = [['control_neg','control_pos'],['mdd_neg','mdd_pos']]
    ax = fig.add_subplot(111)
    ax.set_title("z_0");
    for j in range(n_class):
        for k in range(2):
            ax.scatter(z_0[(c_idx==j)*(m_idx==k),0],z_0[(c_idx==j)*(m_idx==k),1],
                           color = colorss[j][k],
                           s = 5,
                           label = labelss[j][k])
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            right=False, left=False, labelleft=False) # labels along the bottom edge are off
    ax.legend()
    fig.savefig(fig_PATH + "q_z_0.pdf")
    zs = dmfa.q_z_mu.detach().numpy()
    for j in range(1, T, T//5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("z_%d" %j);
        for k in range(n_class):
            for i in range(2):
                ax.scatter(zs[(c_idx==k)*(m_idx==i),j-1,0],zs[(c_idx==k)*(m_idx==i),j-1,1],
                              color = colorss[k][i],
                              s = 5,
                              label = labelss[k][i])
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
        ax.legend()
        fig.savefig(fig_PATH + "q_z_%d.pdf" %j)
    
    zss = np.concatenate((np.expand_dims(z_0, 1), zs), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Latent Space Trajectory");
    for k in range(n_class):
        k_idxs = np.where(c_idx==k)[0]
        k_idxs = np.random.permutation(k_idxs)[:3]
        colorss = np.zeros((len(k_idxs), 3))
        c_id = [2,0,1]
        colorss[:,c_id[k]] = np.arange(0.5, 1, 0.5/len(k_idxs))[:len(k_idxs)]
        for jj, j in enumerate(k_idxs):
            ax.scatter(zss[j,::T//4,0],zss[j,::T//4,1],
                       s=np.array([10,30,45,70])*2, color=colorss[jj])
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
    fig.savefig(fig_PATH + "trajectory.pdf")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("real vs recon fMRI for time point 0");
    ax.plot(mini_batch[0,0,:].detach().numpy())
    ax.plot(y_hat[0,0,:].detach().numpy())
    fig.savefig(fig_PATH + "recon.png")
    
    v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
    v = v.astype("int")
    img = np.zeros(image_dims)
    img[v[:,0],v[:,1],v[:,2]] = y_hat[0,0].detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("sample mid-slice for time point 0");
    ax.imshow(img[:,:,img.shape[-2]//2])
    fig.savefig(fig_PATH + "mid_slice.png")
    plt.close('all')
    
    print("Blob centers are:")
    print(dmfa.q_F_loc_mu + torch.Tensor([i/2 for i in image_dims]))
    
    print("Blob width are:")
    print((dmfa.q_F_scale_mu.exp()/2).sqrt())

    
    v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
    v = v.astype("int")
    img_recon = np.zeros(tuple(image_dims)+(y_hat.shape[1],))
    img_data = np.zeros(tuple(image_dims)+(mini_batch.shape[1],))
    
    for i in range(img_recon.shape[-1]):
        img_recon[v[:,0],v[:,1],v[:,2],i] = y_hat[0].reshape(img_recon.shape[-1],-1).detach().numpy()[i]
        img_data[v[:,0],v[:,1],v[:,2],i] = mini_batch[0].reshape(img_data.shape[-1],-1).detach().numpy()[i]
    
    NIFTI_recon = nib.Nifti1Image(img_recon, np.eye(4))
    NIFTI_data = nib.Nifti1Image(img_data, np.eye(4))
    NIFTI_recon.to_filename(NIFTI_PATH + 'fMRI_recon.nii')
    NIFTI_data.to_filename(NIFTI_PATH + 'fMRI_data.nii')
    
        
    orig = mini_batch[0].detach().numpy()
    recon = y_hat[0].detach().numpy()
    errs = [np.sqrt((((orig[:,i]-orig[:,i].mean())/orig[:,i].std()
            -(recon[:,i]-recon[:,i].mean())/recon[:,i].std())**2).mean())
            for i in range(len(voxel_locations))]
    errs = np.asarray(errs)
    min_err = np.argsort(errs)
    
    for i in min_err[:2]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        orig = mini_batch[0][:,i].detach().numpy()
        recon = y_hat[0][:,i].detach().numpy()
        ax.plot((orig-orig.mean())/orig.std(), label='Real')
        ax.plot((recon-recon.mean())/recon.std(), label='Recovered')
        corrcoef = np.corrcoef(orig, recon)[0,1]
        plt.title('fMRI Time Series corrcoef: %.2f' %corrcoef)
        ax.legend()
        fig.savefig(fig_PATH + "time%d.pdf" %i)
        plt.close('all')



def plot_result_depression_23(dmfa, classes,
                              fig_PATH, mini_batch, y_hat,
                              NIFTI_PATH, exp_type):
    n_class = dmfa.p_c.size(-1)
    T = dmfa.q_w_mu.size(1)
    voxel_locations = dmfa.voxl_locs
    image_dims = dmfa.im_dims
    if exp_type == "depression_2":
        plt_name = "Major Depressive Disorder"
    elif exp_type == "depression_3":
        plt_name = "Control"
        
    print("class Assignments are:")
    print(torch.argmax(dmfa.q_c, dim = 1))
    
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    fig = plt.figure()
    c_idx = classes.detach().numpy()
    m_idx = np.array([1,1,1,1,1,1,1,1,
                      2,2,2,2,2,2,2,2,
                      3,3,3,3,3,3,3,3]*5)
    run_id = 1
    
    colorss = [['cyan','steelblue'],['salmon','orangered'],['orchid','darkmagenta']]
    labelss = [['run1-','run1+'],['run2-','run2+'],['run3-','run3+']]
    ax = fig.add_subplot(111)
    ax.set_title(plt_name + " z_0");
    for k in [run_id-1]:
        for j in range(n_class):
            ax.scatter(z_0[(c_idx==j)*(m_idx==k+1),0],z_0[(c_idx==j)*(m_idx==k+1),1],
                           color = colorss[k][j],
                           s = 5,
                           label = labelss[k][j])
    for k in range(5):
        z_0_temp = z_0[(run_id-1)*8+k*24:(run_id-1)*8+k*24+8]
        c_idx_temp = c_idx[(run_id-1)*8+k*24:(run_id-1)*8+k*24+8]
        for j in range(n_class):
            c_mean = z_0_temp[c_idx_temp==j].mean(axis=0)
            c_std = z_0_temp[c_idx_temp==j].std(axis=0)
            circle = Ellipse((c_mean[0], c_mean[1]), c_std[0]*2, c_std[1]*2, color=colorss[run_id-1][j], alpha = 0.2)
            ax.add_artist(circle)
    
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            right=False, left=False, labelleft=False) # labels along the bottom edge are off
    ax.legend()
    fig.savefig(fig_PATH + "run%d-q_z_0.pdf" %run_id)
    zs = dmfa.q_z_mu.detach().numpy()
    for j in range(1, T, T//5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(plt_name + " z_%d" %j);
        for i in [run_id-1]:
            for k in range(n_class):
                ax.scatter(zs[(c_idx==k)*(m_idx==i+1),j-1,0],zs[(c_idx==k)*(m_idx==i+1),j-1,1],
                              color = colorss[i][k],
                              s = 5,
                              label = labelss[i][k])
        
        for k in range(5):
            z_0_temp = zs[(run_id-1)*8+k*24:(run_id-1)*8+k*24+8,j-1]
            c_idx_temp = c_idx[(run_id-1)*8+k*24:(run_id-1)*8+k*24+8]
            for i in range(n_class):
                c_mean = z_0_temp[c_idx_temp==i].mean(axis=0)
                c_std = z_0_temp[c_idx_temp==i].std(axis=0)
                circle = Ellipse((c_mean[0], c_mean[1]), c_std[0]*2, c_std[1]*2, color=colorss[run_id-1][i], alpha = 0.2)
                ax.add_artist(circle)
        
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
        ax.legend()
        fig.savefig(fig_PATH + "run%d-q_z_%d.pdf" %(run_id,j))
    
    zss = np.concatenate((np.expand_dims(z_0, 1), zs), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Latent Space Trajectory");
    for k in range(n_class):
        k_idxs = np.where(c_idx==k)[0]
        k_idxs = np.random.permutation(k_idxs)[:3]
        colorss = np.zeros((len(k_idxs), 3))
        c_id = [2,0,1]
        colorss[:,c_id[k]] = np.arange(0.5, 1, 0.5/len(k_idxs))[:len(k_idxs)]
        for jj, j in enumerate(k_idxs):
            ax.scatter(zss[j,::T//4,0],zss[j,::T//4,1],
                       s=np.array([10,30,45,70])*2, color=colorss[jj])
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
    fig.savefig(fig_PATH + "trajectory.pdf")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("real vs recon fMRI for time point 0");
    ax.plot(mini_batch[0,0,:].detach().numpy())
    ax.plot(y_hat[0,0,:].detach().numpy())
    fig.savefig(fig_PATH + "recon.png")
    
    v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
    v = v.astype("int")
    img = np.zeros(image_dims)
    img[v[:,0],v[:,1],v[:,2]] = y_hat[0,0].detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("sample mid-slice for time point 0");
    ax.imshow(img[:,:,img.shape[-2]//2])
    fig.savefig(fig_PATH + "mid_slice.png")
    plt.close('all')
    
    print("Blob centers are:")
    print(dmfa.q_F_loc_mu + torch.Tensor([i/2 for i in image_dims]))
    
    print("Blob width are:")
    print((dmfa.q_F_scale_mu.exp()/2).sqrt())

    
    v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
    v = v.astype("int")
    img_recon = np.zeros(tuple(image_dims)+(y_hat.shape[1],))
    img_data = np.zeros(tuple(image_dims)+(mini_batch.shape[1],))
    
    for i in range(img_recon.shape[-1]):
        img_recon[v[:,0],v[:,1],v[:,2],i] = y_hat[0].reshape(img_recon.shape[-1],-1).detach().numpy()[i]
        img_data[v[:,0],v[:,1],v[:,2],i] = mini_batch[0].reshape(img_data.shape[-1],-1).detach().numpy()[i]
    
    NIFTI_recon = nib.Nifti1Image(img_recon, np.eye(4))
    NIFTI_data = nib.Nifti1Image(img_data, np.eye(4))
    NIFTI_recon.to_filename(NIFTI_PATH + 'fMRI_recon.nii')
    NIFTI_data.to_filename(NIFTI_PATH + 'fMRI_data.nii')
    
        
    orig = mini_batch[0].detach().numpy()
    recon = y_hat[0].detach().numpy()
    errs = [np.sqrt((((orig[:,i]-orig[:,i].mean())/orig[:,i].std()
            -(recon[:,i]-recon[:,i].mean())/recon[:,i].std())**2).mean())
            for i in range(len(voxel_locations))]
    errs = np.asarray(errs)
    min_err = np.argsort(errs)
    
    for i in min_err[:2]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        orig = mini_batch[0][:,i].detach().numpy()
        recon = y_hat[0][:,i].detach().numpy()
        ax.plot((orig-orig.mean())/orig.std(), label='Real')
        ax.plot((recon-recon.mean())/recon.std(), label='Recovered')
        corrcoef = np.corrcoef(orig, recon)[0,1]
        plt.title('fMRI Time Series corrcoef: %.2f' %corrcoef)
        ax.legend()
        fig.savefig(fig_PATH + "time%d.pdf" %i)
        plt.close('all')
        
        
    ### New Approach
    fig = plt.figure()
    colorss = [['cyan','steelblue'],['salmon','orangered'],['orchid','darkmagenta']]
    labelss = [['negative music','positive music']]
    ax = fig.add_subplot(111)
    ax.set_title(plt_name + " z_0");
    for j in range(n_class):
        ax.scatter(z_0[c_idx==j,0],z_0[c_idx==j,1],
                       color = colorss[0][j],
                       s = 5,
                       label = labelss[0][j])
    for k in range(5):
        z_0_temp = z_0[k*24:(k+1)*24]
        c_idx_temp = c_idx[k*24:(k+1)*24]
        for j in range(n_class):
            c_mean = z_0_temp[c_idx_temp==j].mean(axis=0)
            c_std = z_0_temp[c_idx_temp==j].std(axis=0)
            circle = Ellipse((c_mean[0], c_mean[1]), c_std[0]*2, c_std[1]*2, color=colorss[0][j], alpha = 0.2)
            ax.add_artist(circle)
    
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            right=False, left=False, labelleft=False) # labels along the bottom edge are off
    ax.legend()
    fig.savefig(fig_PATH + "qq_z_0.pdf")
    zs = dmfa.q_z_mu.detach().numpy()
    for j in range(1, T, T//5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(plt_name + " z_%d" %j);
        for k in range(n_class):
            ax.scatter(zs[c_idx==k,j-1,0],zs[c_idx==k,j-1,1],
                          color = colorss[0][k],
                          s = 5,
                          label = labelss[0][k])
        
        for k in range(5):
            z_0_temp = zs[k*24:(k+1)*24,j-1]
            c_idx_temp = c_idx[k*24:(k+1)*24]
            for i in range(n_class):
                c_mean = z_0_temp[c_idx_temp==i].mean(axis=0)
                c_std = z_0_temp[c_idx_temp==i].std(axis=0)
                circle = Ellipse((c_mean[0], c_mean[1]), c_std[0]*2, c_std[1]*2, color=colorss[0][i], alpha = 0.2)
                ax.add_artist(circle)
        
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
        ax.legend()
        fig.savefig(fig_PATH + "qq_z_%d.pdf" %j)
    plt.close('all')



def plot_result(dmfa, classes,
                fig_PATH, mini_batch, y_hat,
                NIFTI_PATH = None, exp_type = "custom",
                train = False, prefix = '', ext = ".pdf",
                root_dir = None):
    exp_type = exp_type.split('_')
    n_class = dmfa.p_c.size(-1)
    T = dmfa.q_w_mu.size(1)
    voxel_locations = dmfa.voxl_locs
    image_dims = dmfa.im_dims
    if not train:
        print("class Assignments are:")
        print(torch.argmax(dmfa.q_c, dim = 1))
    
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    z_0_p = dmfa.z_0_mu.detach().numpy()
    z_0_p_sig = dmfa.z_0_sig.exp().detach().numpy()
    if exp_type[0] == "autism":
        labels = ['Autism','Control']
    else:
        labels = ['group%d'%(c+1) for c in range(n_class)]

    acc = torch.sum(dmfa.q_c.argmax(dim=1)==classes).float()/dmfa.q_c.shape[0]
    print('Accuracy of Clustering: %0.2f' %(max(acc,1-acc)))
    fig = plt.figure()
    colors = ['r','b','g','y']
    c_idx = classes.detach().numpy()
    ax = fig.add_subplot(111)
    ax.set_title("z_0");
    for j in range(n_class):
        ax.scatter(z_0[c_idx==j,0],z_0[c_idx==j,1], label = labels[j])
        circle = Ellipse((z_0_p[j, 0], z_0_p[j, 1]),
                         z_0_p_sig[j,0]*2, z_0_p_sig[j,1]*2,
                         color=colors[j], alpha = 0.2)
        ax.add_artist(circle)
    ax.legend()
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            right=False, left=False, labelleft=False) # labels along the bottom edge are off
    fig.savefig(fig_PATH + "%sq_z_0" %prefix + ext)
    zs = dmfa.q_z_mu.detach().numpy()
    for j in range(1, T, T//5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("z_%d" %j);
        for k in range(n_class):
            ax.scatter(zs[c_idx==k,j-1,0],zs[c_idx==k,j-1,1], label = labels[k])
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
        ax.legend()
        fig.savefig(fig_PATH + "%sq_z_%d" %(prefix,j) + ext)
    
    zss = np.concatenate((np.expand_dims(z_0, 1), zs), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Latent Space Trajectory");
    if exp_type[0] == "autism":
        labels = [['Autism/1','Autism/2'],['Control/1','Control/2']]
    else:
        labels = [['group1/1','group1/2'],['group2/1','group2/2'],['group3/1','group3/2']]
    for k in range(min(n_class,3)): #plot at most for three clusters
        k_idxs = np.where(c_idx==k)[0]
        k_idxs = k_idxs[:2]
        colorss = np.zeros((len(k_idxs), 3))
        c_id = [2,0,1]
        colorss[:,c_id[k]] = np.arange(0.5, 1, 0.5/len(k_idxs))[:len(k_idxs)]
        for jj, j in enumerate(k_idxs):
            ax.scatter(zss[j,::T//4,0],zss[j,::T//4,1],
                       s=np.array([10,30,45,70])*2, color=colorss[jj], label = labels[k][jj])
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        right=False, left=False, labelleft=False) # labels along the bottom edge are off
    ax.legend()
    fig.savefig(fig_PATH + "%strajectory" %prefix + ext)
    
    if exp_type[0] != "depression":
        z_p = dmfa.z_0_mu.detach().numpy()
        z_0s = np.expand_dims(z_p,1)
        z_0sigs = np.expand_dims(z_0_p_sig,1)
        for j in range(1, T, 1):
            z_p, z_p_sig = dmfa.trans(torch.Tensor(z_p), None)
            z_p = z_p.detach().numpy()
            z_0s = np.concatenate((z_0s, np.expand_dims(z_p, 1)), axis = 1)
            z_0sigs = np.concatenate((z_0sigs, np.expand_dims(z_p_sig.exp().detach().numpy(), 1)), axis = 1)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Mean Latent Space Trajectory", fontsize=12);
        for k in range(n_class):
            ax.quiver(z_0s[k,:-1:T//5,0],z_0s[k,:-1:T//5,1],
                      z_0s[k,1::T//5,0]-z_0s[k,:-1:T//5,0], z_0s[k,1::T//5,1]-z_0s[k,:-1:T//5,1],
                      color=colors[k], #label = labels[k],
                      scale_units='xy', angles='xy', scale=0.1)
            for j in range(T//5, T, T//5):
                circle = Ellipse((z_0s[k,j, 0], z_0s[k,j, 1]),
                                     z_0sigs[k,j,0]*2, z_0sigs[k,j,1]*2,
                                     color=colors[k], alpha = 0.2)
                ax.add_artist(circle)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            right='off', left='off', labelleft='off') # labels along the bottom edge are off
        fig.savefig(fig_PATH + "%strajectory_mean" %prefix + ext)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("real vs recon for time point 0");
    ax.plot(mini_batch[0,0,:].detach().numpy())
    ax.plot(y_hat[0,0,:].detach().numpy())
    fig.savefig(fig_PATH + "%srecon.png" %prefix)
    
    v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
    v = v.astype("int")
    img = np.zeros(image_dims)
    img[v[:,0],v[:,1],v[:,2]] = y_hat[0,0].detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("sample mid-slice for time point 0");
    ax.imshow(img[:,:,img.shape[-2]//2])
    fig.savefig(fig_PATH + "%smid_slice.png" %prefix)
    plt.close('all')
    
    if not train:
        print("Blob centers are:")
        print(dmfa.q_F_loc_mu + torch.Tensor([i/2 for i in image_dims]))
        
        print("Blob width are:")
        print((dmfa.q_F_scale_mu.exp()/2).sqrt())
    
        
        v = voxel_locations.detach().numpy()+[i/2 for i in image_dims]
        v = v.astype("int")
        img_recon = np.zeros(tuple(image_dims)+(y_hat.shape[1],))
        img_data = np.zeros(tuple(image_dims)+(mini_batch.shape[1],))
        
        for i in range(img_recon.shape[-1]):
            img_recon[v[:,0],v[:,1],v[:,2],i] = y_hat[0].reshape(img_recon.shape[-1],-1).detach().numpy()[i]
            img_data[v[:,0],v[:,1],v[:,2],i] = mini_batch[0].reshape(img_data.shape[-1],-1).detach().numpy()[i]
        
        NIFTI_recon = nib.Nifti1Image(img_recon, np.eye(4))
        NIFTI_data = nib.Nifti1Image(img_data, np.eye(4))
        NIFTI_recon.to_filename(NIFTI_PATH + 'fMRI_recon.nii')
        NIFTI_data.to_filename(NIFTI_PATH + 'fMRI_data.nii')
            
        orig = mini_batch[0].detach().numpy()
        recon = y_hat[0].detach().numpy()
        errs = [np.sqrt((((orig[:,i]-orig[:,i].mean())/orig[:,i].std()
                -(recon[:,i]-recon[:,i].mean())/recon[:,i].std())**2).mean())
                for i in range(len(voxel_locations))]
        errs = np.asarray(errs)
        min_err = np.argsort(errs)
        
        for i in min_err[:2]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            orig = mini_batch[0][:,i].detach().numpy()
            recon = y_hat[0][:,i].detach().numpy()
            ax.plot((orig-orig.mean())/orig.std(), label='Real')
            ax.plot((recon-recon.mean())/recon.std(), label='Recovered')
            corrcoef = np.corrcoef(orig, recon)[0,1]
            plt.title('fMRI Time Series corrcoef: %.2f' %corrcoef)
            ax.legend()
            fig.savefig(fig_PATH + "time%d.pdf" %i)
            plt.close('all')
        
        #######################################
        #SVM Classification for Autism Dataset
        ######################################
        if exp_type[0] == "autism":
            ws = dmfa.q_w_mu.detach().numpy()
            if True:
                if len(exp_type) == 1:
                    aut_lens = np.cumsum(np.load(root_dir + 'autism_lens.npy'))
                else:
                    site = '_'.join(exp_type[1:])
                    aut_sites = np.load(root_dir + 'autism_sites.npy')
                    aut_lens = np.cumsum(np.load(root_dir + 'autism_lens.npy')[aut_sites==site])
            
                aut_lens = np.concatenate(([0],aut_lens))
                ws = [ws[aut_lens[i]:aut_lens[i+1]].reshape(-1,dmfa.q_w_mu.size(-1)) for i in range(len(aut_lens)-1)]
                class_data = [classes[aut_lens[i]:aut_lens[i+1]][0] for i in range(len(aut_lens)-1)]
                class_data =np.asarray(class_data, dtype=int)
            else:
                class_data = classes.detach().numpy()                

            train_data = np.asarray([np.corrcoef(sample.T)[np.triu_indices(sample.shape[1], 1)] for sample in ws])
            train_data = np.absolute(train_data)
            clf = [svm.SVC(kernel='linear', C=1), svm.SVC(kernel='rbf', C=1)]
            scores = [[] for _ in range(len(clf))]
            for i in range(50):
                X_train, X_test, y_train, y_test = train_test_split(
                        train_data, class_data, test_size=0.1)
                for j in range(len(clf)):
                    clf[j].fit(X_train, y_train)
                    scores[j].append(clf[j].score(X_test, y_test))
            scores = [np.asarray(x) for x in scores]
            j = np.argmax([x.mean() for x in scores])
            print("Accuracy of SVM (+/-) SE: %.2f (+/-) %.2f" %(scores[j].mean(), scores[j].std()/np.sqrt(50)))
        #######################################
        

def prediction_result(dmfa,
                fig_PATH,
                exp_type = "custom",
                prefix = '',
                ext = ".pdf",
                idxs = None,
                predict=False,
                root_dir = None,
                list_IDs = None):
    T = dmfa.q_w_mu.size(1)
    vxl_num = len(dmfa.voxl_locs)
    
    ws = dmfa.q_w_mu[idxs[-1]].detach().numpy()
    z_t_1 = dmfa.q_z_0_mu[idxs[-1]].reshape(1, -1)
    if exp_type == "depression_5":
        with open(root_dir + 'meta_data.json', 'r') as f:
            meta_data = json.load(f)
        u_t_1 = torch.zeros(1, 1)
        if meta_data['run_type'][list_IDs[idxs[-1]]] == 'm':
            u_t_1[0,:] = 1
    else:
        u_t_1 = None
    for i in range(T):
        p_w_mu, _ = dmfa.temp(z_t_1)
        p_z_mu, _ = dmfa.trans(z_t_1, u_t_1)
        ws = np.concatenate((ws, p_w_mu.detach().numpy()), axis = 0)
        z_t_1 = p_z_mu * 1.0

    f_F = dmfa.RBF_to_Voxel(dmfa.q_F_loc_mu.unsqueeze(0),
                            dmfa.q_F_scale_mu.unsqueeze(0))[0]
    y_pred = np.matmul(ws, f_F.detach().numpy())
    
    orign = np.load(os.path.join(root_dir, 'fMRI_%.5d.npz' %idxs[-1]))['arr_0']
    recons = y_pred[-T:]
    errs = [np.sqrt((((orign[:,i]-orign[:,i].mean())/orign[:,i].std()
            -(recons[:,i]-recons[:,i].mean())/recons[:,i].std())**2).mean())
            for i in range(vxl_num)]
    errs = np.asarray(errs)
    min_err = np.argsort(errs)
    
    for i in min_err[:2]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        orig = orign[:,i]
        recon = y_pred[:T][:,i]
        pred = recons[:,i]
        ax.plot((orig-orig.mean())/orig.std(), label='Real')
        ax.plot((recon-recon.mean())/recon.std(), label='Recovered')
        ax.plot((pred-pred.mean())/pred.std(), label='Predicted')
        corrcoef = np.corrcoef(orig, pred)[0,1]
        plt.title('fMRI Time Series corrcoef: %.2f' %corrcoef)
        ax.legend()
        fig.savefig(fig_PATH + "%stime_pred%d" %(prefix,i)+ext)
        plt.close('all')
     
    if predict:
        
        if exp_type == "depression_5":
            titles = ['Music', 'Nonmusic']
        else:
            titles = ['']
        y_pred_ = [np.array([]).reshape(0, vxl_num) for _ in range(len(titles))]
        orign_ = [np.array([]).reshape(0, vxl_num) for _ in range(len(titles))]
        RMSE_ = [[] for _ in range(len(titles))]
        for index in idxs:
            z_t_1 = dmfa.q_z_0_mu[index].reshape(1, -1)
            p_w_mu, _ = dmfa.temp(z_t_1)
            ws = p_w_mu.detach().numpy()
            if exp_type == "depression_5":
                u_t_1 = torch.zeros(1, 1)
                if meta_data['run_type'][list_IDs[index]] == 'm':
                    u_t_1[0,:] = 1
            else:
                u_t_1 = None
            for i in range(T-1):
                p_z_mu, _ = dmfa.trans(z_t_1, u_t_1)
                p_w_mu, _ = dmfa.temp(p_z_mu)
                ws = np.concatenate((ws, p_w_mu.detach().numpy()), axis = 0)
                z_t_1 = dmfa.q_z_mu[index, i].reshape(1,-1)
            
            y_pred = np.matmul(ws, f_F.detach().numpy())
            orign = np.load(os.path.join(root_dir, 'fMRI_%.5d.npz' %list_IDs[index]))['arr_0']
            
            errs = [np.sqrt(((orign[:,i]-y_pred[:,i])**2).mean()) for i in range(vxl_num)]
            errs = np.asarray(errs)
            min_err = np.argsort(errs)
            
            RMSE = np.sqrt((errs[min_err[:3000]]**2).mean())
            
            cnt = 0
            if exp_type == "depression_5":
                if meta_data['run_type'][list_IDs[index]] == 'n':
                    cnt = 1
            y_pred_[cnt] = np.concatenate((y_pred_[cnt],y_pred), axis=0)
            orign_[cnt] = np.concatenate((orign_[cnt],orign), axis=0)
            RMSE_[cnt].append(RMSE)
        for i in range(len(titles)):
            print('RMSE (top 5 per datapoint) %s' %titles[i], RMSE_[i])
    
        y_pred_norm = y_pred_ #[(x-x.mean())/x.std() for x in y_pred_]
        orign_norm = orign_ #[(x-x.mean())/x.std() for x in orign_]

        RMSE_ = [np.sqrt(np.power(y_pred_norm[i] - orign_norm[i],2).mean()) for i in range(len(titles))]
        MAPE_ = [RMSE_[i]/np.sqrt(np.power(orign_norm[i],2).mean())*100 for i in range(len(titles))]
        for i in range(len(titles)):
            print('RMSE %s: %.8f' %(titles[i],RMSE_[i]))
            print('MAPE %s: %.8f' %(titles[i],MAPE_[i]))
        
        cnt = 0
        for orign, recons in [(orign_[i], y_pred_[i]) for i in range(len(titles))]:

            errs = [np.sqrt(((orign[:,i]-recons[:,i])**2).mean()) for i in range(vxl_num)]
            errs = np.asarray(errs)
            min_err = np.argsort(errs)
            
            RMSE = np.sqrt((errs[min_err[:3000]]**2).mean())
            print('RMSE (top 5) %s: %.8f' %(titles[cnt],RMSE))
            scale = (orign[:,min_err[:3000]]- orign[:,min_err[:3000]].mean())/ orign[:,min_err[:3000]].std()
            MAPE = RMSE/np.sqrt(np.power(scale,2).mean())*100
            print('MAPE (top 5) %s: %.8f' %(titles[cnt],MAPE))
        
            errs_corr = [np.sqrt((((orign[:,i]-orign[:,i].mean())/orign[:,i].std()
                    -(recons[:,i]-recons[:,i].mean())/recons[:,i].std())**2).mean())
                    for i in range(vxl_num)]
            errs_corr = np.asarray(errs_corr)
            min_err_corr = np.argsort(errs_corr)
            
            for i in min_err_corr[:5]:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                orig = orign[:,i]
                pred = recons[:,i]
                ax.plot((orig-orig.mean())/orig.std(), label='Actual')
                ax.plot((pred-pred.mean())/pred.std(), 'r-',label='Predicted')
                corrcoef = np.corrcoef(orig, pred)[0,1]
                plt.title('%s fMRI Time Series corrcoef: %.2f' %(titles[cnt],corrcoef))
                ax.legend()
                ax.set_xlabel('Time', fontsize=16)
                fig.savefig(fig_PATH + "%stime_pred_%s%d" %(prefix,titles[cnt],i)+ext)
                plt.close('all')
            cnt += 1