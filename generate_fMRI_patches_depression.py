

import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
import pdb
from detect_local_minima import *
import scipy.signal
import json
import argparse

patch_idxs = [[0,6],
             [6,12],
            [12,18],
            [18,24],
            [24,30],
            [29,35],
            [35,41],
            [41,47],
            [47,53],
            [52,58],
            [58,64],
            [64,70],
            [70,76],
            [75,81],
            [81,87],
            [87,93],
            [93,99],
            [99,105]]

def get_patches(data, avg, std, num, patch_len, u_dim, save_path,
                return_flag = True,
                save_flag = False):
    if return_flag:
        patches = [((torch.FloatTensor(data[idx[0]:idx[1]]-avg)/std),
                    torch.zeros((patch_len, u_dim)),
                    torch.LongTensor([n+num]))
                   for n, idx in enumerate(patch_idxs)]
    else: 
        patches = [] 
    
    if save_flag:
        [[np.savez_compressed(save_path + "fMRI_%.5d.npz" %(n+num),
                              (data[idx[0]:idx[1]]-avg)/std),
          np.savez_compressed(save_path + "stimuli_%.5d.npz" %(n+num),
                              np.zeros((patch_len, u_dim)))] #by default no stimuli is present
        for n, idx in enumerate(patch_idxs)]

    num += len(patch_idxs)
    return patches, num
    
        
def load_data(data_path = 'S:/Users/fMRI/datasets/',
              save_path = 'S:/Users/fMRI/Data/',
              patch_len = 6,
              u_dim  = 1,
              return_flag = True,
              save_flag = False):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_list = os.listdir(data_path)
    data_folders = [data_list[i] for i in range(len(data_list)) if os.path.isdir(os.path.join(data_path,data_list[i]))]
    
    
    std = 0
    avg = 0
    num = 0
    data_avg = 0
    mask = 1
    for i in range(len(data_folders)):
        print('Folder %d selected' %(i+1))
        data_dir = os.path.join(data_path, data_folders[i], 'func')
        files_dir = glob(data_dir + '/*music*preproc*.gz')
        masks_dir = glob(data_dir + '/*music*brainmask*.gz')
        for j in range(len(files_dir)):
            print(files_dir[j])
            img = nib.load(files_dir[j])
            data = img.get_fdata()
            im = nib.load(masks_dir[j])
            mask_data = im.get_fdata()           
            z_idxs = np.where(mask_data==0)
            mask *= mask_data
            print(len(z_idxs[0]))
            data_avg += data.mean(axis = 3)
            std += data.std()
            avg += data.mean() 
            num += 1
    
    std = std/num
    avg = avg/num
    print('avg = %.2f, std = %.2f' %(avg, std))
    
    data_avg[mask==0] = 0
    v_i, v_j, v_k = data_avg.shape
    print('# of non-zeros voxels: %d' %len(np.where(mask != 0)[0]))
    mask_idxs = np.where(mask != 0)
    voxel_locations = np.asarray(mask_idxs).T - [v_i/2, v_j/2, v_k/2]
    maxima_locs = np.asarray(detect_local_minima(scipy.signal.medfilt(data_avg, kernel_size = 5))).T - [v_i/2, v_j/2, v_k/2]
    image_dims = (v_i,v_j,v_k)
    if save_flag:
        np.savez_compressed(save_path+ "image_dims.npz", image_dims)
        np.savez_compressed(save_path+ "maxima_locs.npz", maxima_locs)
        np.savez_compressed(save_path+ "vox_locs.npz", voxel_locations)
    
    if return_flag:
        voxel_locations = torch.FloatTensor(voxel_locations)
    
    training_set = []
    num = 0
    
    c_group = [[1,-1],[2,1],[3,1],[4,-1],[5,1],[6,-1],[7,-1],[8,1],[9,-1],[10,1],[11,-1],
               [12,1],[13,1],[14,1],[15,-1],[16,-1],[17,-1],[18,-1],[19,1],[20,1]]
    m_group = [[1,1],[2,-1],[3,-1],[4,1],[5,-1],[6,1],[7,-1],[8,1],[9,-1],[10,-1],[11,1],
               [12,1],[13,1],[14,-1],[15,-1],[16,1],[17,1],[18,-1],[19,1]]
               
    labels = [0,0,1,1,0,0,-1,-1,0,0,1,1,0,0,-1,-1,0,0] #start with positive
               
    meta_data = {}
    meta_data['subj_id'] = []
    meta_data['subj_group'] = []
    meta_data['run_id'] = []
    meta_data['run_type'] = []
    meta_data['music_type'] = []
        
    # each datapoint for shuffling is a tuple ((T*V), (T*u_dim), n)
    for i in range(len(data_folders)):
        print('Folder %d selected' %(i+1))
        subj_id = int(data_folders[i][-2:])
        subj_group = data_folders[i][4]
        data_dir = os.path.join(data_path, data_folders[i], 'func')
        files_dir = glob(data_dir + '/*music*preproc*.gz')
        for j in range(len(files_dir)):
            print(files_dir[j])
            run_id = int(os.path.split(files_dir[j])[-1][-49])
            if subj_group == 'c':
                run_type = os.path.split(files_dir[j])[-1][19]
                music_type = -(-1)**run_id*c_group[subj_id-1][1]*labels
            else:
                run_type = os.path.split(files_dir[j])[-1][16]
                music_type = -(-1)**run_id*m_group[subj_id-1][1]*labels
            
            meta_data['subj_id'] += [subj_id]*len(patch_idxs)
            meta_data['subj_group'] += [subj_group]*len(patch_idxs)
            meta_data['run_id'] += [run_id]*len(patch_idxs) 
            meta_data['run_type'] += [run_type]*len(patch_idxs)
            meta_data['music_type'] += music_type
            
            img = nib.load(files_dir[j])
            data = img.get_fdata()
            data = data[mask_idxs].T
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
    
    if save_flag:
        json_file = json.dumps(meta_data)
        f = open(save_path+"meta_data.json","w")
        f.write(json_file)
        f.close()
    # to read json file:
    ##with open(save_path+'meta_data.json', 'r') as f:
    ##    meta_data = json.load(f)
            
    if return_flag:    
        return image_dims, voxel_locations, maxima_locs, training_set, meta_data        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-du', '--u_dim', type=int, default=0)
    parser.add_argument('-T', '--patch_len', type=int, default=6)
    parser.add_argument('-dir', '--data_path', type=str, default='./fmriprep/')
    parser.add_argument('-spath', '--save_path', type=str, default='./data_music/')
    args = parser.parse_args()
    
    load_data(data_path = args.data_path,
              save_path = args.save_path,
              patch_len = args.patch_len,
              u_dim  = args.u_dim,
              return_flag = False,
              save_flag = True)