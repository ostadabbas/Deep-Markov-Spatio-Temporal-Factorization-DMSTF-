
import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
import pdb
from detect_local_minima import *
import scipy.signal
import argparse


def get_patches(data, avg, std, num, patch_len, u_dim, save_path,
                return_flag = True,
                save_flag = False):
    
    if return_flag:
        patches = [((torch.FloatTensor(data[patch_len*n:patch_len*(n+1)]-avg)/std),
                    torch.zeros((patch_len, u_dim)),
                    torch.LongTensor([n+num]))
                   for n in range(len(data)//patch_len)]
    else: 
        patches = [] 
    
    if save_flag:
        [[np.savez_compressed(save_path + "fMRI_%.5d.npz" %(n+num),
                              (data[patch_len*n:patch_len*(n+1)]-avg)/std),
          np.savez_compressed(save_path + "stimuli_%.5d.npz" %(n+num),
                              np.zeros((patch_len, u_dim)))]
        for n in range(len(data)//patch_len)]

    num += len(data)//patch_len
    return patches, num
    
        
def load_data(data_path = 'S:/Users/fMRI/datasets/',
              save_path = 'S:/Users/fMRI/Data/',
              patch_len = 10,
              u_dim  = 1,
              return_flag = True,
              save_flag = False,
              ext = '/Cal*.gz'):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_list = os.listdir(data_path)
    data_folders = [data_list[i] for i in range(len(data_list)) if os.path.isdir(os.path.join(data_path,data_list[i]))]
    
    
    std = 0
    avg = 0
    num = 0
    data_avg = 0
    mask = 1
    classes = []
    for i in range(len(data_folders)):
        print('Folder %s selected' %data_folders[i])
        data_dir = os.path.join(data_path, data_folders[i])
        files_dir = glob(data_dir + ext)
        for j in range(len(files_dir)):
            img = nib.load(files_dir[j])
            data = img.get_fdata()
            classes += [i]*(data.shape[-1]//patch_len)
            data[np.isnan(data)] = 0
            z_idxs = np.where(data.mean(axis = 3)==0)
            mask_data = np.ones(data.shape[:3])
            mask_data[z_idxs] = 0
            mask *= mask_data
            print('# of zero voxels_%d: %d' %(j, len(z_idxs[0])))
            data_avg += data.mean(axis = 3)
            std += data.std()
            avg += data.mean() 
            num += 1
    
    std = std/num
    avg = avg/num
    print('avg = %.2f, std = %.2f' %(avg, std))
#    
    data_avg[mask==0] = 0
    v_i, v_j, v_k = data_avg.shape
    idxs = np.where(mask != 0)
    print('# of non-zero voxels: %d' %len(np.where(mask != 0)[0]))
    voxel_locations = np.asarray(idxs).T - [v_i/2, v_j/2, v_k/2]
    maxima_locs = np.asarray(detect_local_minima(scipy.signal.medfilt(data_avg, kernel_size = 5))).T - [v_i/2, v_j/2, v_k/2]
    image_dims = (v_i,v_j,v_k)
    if save_flag:
        np.savez_compressed(save_path+ "vox_locs.npz", voxel_locations)
        np.savez_compressed(save_path+ "image_dims.npz", image_dims)
        np.savez_compressed(save_path+ "maxima_locs.npz", maxima_locs)
        np.savez_compressed(save_path+ "classes.npz", classes)
    
    if return_flag:
        voxel_locations = torch.FloatTensor(voxel_locations)
    
    training_set = []
    num = 0
    # each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)
    for i in range(len(data_folders)):
        print('Folder %s selected' %data_folders[i])
        data_dir = os.path.join(data_path, data_folders[i])
        files_dir = glob(data_dir + ext)
        for j in range(len(files_dir)):
            img = nib.load(files_dir[j])
            data = img.get_fdata()
            data = data[idxs].T
            avg = data.mean()
            std = data.std()
            patches, num = get_patches(data,
                                       avg,
                                       std,
                                       num,
                                       patch_len,
                                       u_dim,
                                       save_path,
                                       return_flag,
                                       save_flag)
            training_set += patches
        
    classes = torch.LongTensor(classes)
    #by default data in each subfolder under data path is considered to be in similar class
    # if you wish to assign classes differently, simply discard classes, and assign it manually
    # in main code for training
    if return_flag:    
        return image_dims, voxel_locations, maxima_locs, training_set, classes        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-du', '--u_dim', type=int, default=0)
    parser.add_argument('-T', '--patch_len', type=int, default=5)
    parser.add_argument('-dir', '--data_path', type=str, default='./data/')
    parser.add_argument('-ext', '--files_extension', type=str, default='/Cal*.gz')
    parser.add_argument('-spath', '--save_path', type=str, default='./data_prep/')
    args = parser.parse_args()
    
    load_data(data_path = args.data_path,
              save_path = args.save_path,
              patch_len = args.patch_len,
              u_dim  = args.u_dim,
              return_flag = False,
              save_flag = True,
              ext = args.files_extension)