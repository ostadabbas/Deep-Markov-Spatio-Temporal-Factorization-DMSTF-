
import json
import numpy as np
import torch
from glob import glob

def load_data(root_dir, exp_type):
    
    exp_type = exp_type.split('_')
    
    if exp_type[0] == "depression":
        
        # to read json file:
        with open(root_dir + 'meta_data.json', 'r') as f:
            meta_data = json.load(f)
        '''
        meta_data['subj_id']
        meta_data['subj_group']
        meta_data['run_id']
        meta_data['run_type']
        meta_data['music_type']
        '''
        ### data selection
        if exp_type[1] == "1":
            att_list = [('subj_id', [1,2,3,4,5,6,7,8,9,10]),
                        ('subj_group', ['c','m']),
                        ('run_type', ['m']),
                        ('music_type', [1, -1])]
        elif exp_type[1] == "2":
            att_list = [('subj_id', [1,2,3,4,5]),
                        ('subj_group', ['m']),
                        ('run_type', ['m']),
                        ('music_type', [1, -1])]
        elif exp_type[1] == "3":    
            att_list = [('subj_id', [1,2,3,4,5]),
                     ('subj_group', ['c']),
                     ('run_type', ['m']),
                     ('music_type', [1, -1])]
        elif exp_type[1] == "4":
            att_list = [('run_type', ['m']),
                        ('music_type', [1, -1])]
        elif exp_type[1] == "5":
            att_list = [('subj_id', [1]),
                        ('subj_group', ['m']),
                        ('music_type', [1, -1])]
            
        mask = 1
        for i in range(len(att_list)):
            mask_att = np.zeros(len(meta_data[att_list[i][0]]))
            for j in range(len(meta_data[att_list[i][0]])):
                if meta_data[att_list[i][0]][j] in att_list[i][1]:
                    mask_att[j] = 1
            mask *= mask_att
        indices = np.where(mask==1)
        
        if exp_type[1] in ["1", "4"]:
            classes_0 = [meta_data['subj_group'][i] for i in indices[0]]
            classes = [0]*len(classes_0)
            for i in range(len(classes_0)):
                if classes_0[i] == 'm':
                    classes[i] = 1
            classes = torch.LongTensor(classes)
        elif exp_type[1] in ["2", "3"]:
            classes = torch.LongTensor(meta_data['music_type'])[indices] + 1
            classes[classes==2] = 1
        elif exp_type[1] == "5":
            classes = torch.zeros(len(indices[0]), dtype = torch.int64)
        list_IDs = indices[0]
    elif exp_type[0] == "synthetic":
        list_IDs = np.array([i for i in range(30)])
        classes = torch.LongTensor([0]*10 + [1]*10 + [2]*10)
    else:
        classes = np.load(root_dir + 'classes.npz')['arr_0']
        classes = torch.LongTensor(classes)
    
    if exp_type[0] == "autism":
        aut_lens = np.load(root_dir + 'autism_lens.npy')
        aut_sites = np.load(root_dir + 'autism_sites.npy')
        list_IDs = np.array([i for i in range(int(aut_lens.sum()))])
        aut_sites = [[aut_sites[i]]*aut_lens[i] for i in range(len(aut_sites))]
        aut_sites = np.asarray([i for j in aut_sites for i in j])
        if len(exp_type) > 1:
            site = '_'.join(exp_type[1:])
            list_IDs = np.where(aut_sites == site)[0]
            classes = classes[np.where(aut_sites == site)]
        
    if exp_type[0] == "custom":
        list_IDs = np.array([i for i in range(len(glob(root_dir + 'fMRI_*.npz')))])
        
    ####
    voxel_locations = torch.FloatTensor(np.load(root_dir + 'vox_locs.npz')['arr_0'])
    image_dims = np.load(root_dir + 'image_dims.npz')['arr_0']
    maxima_locs = np.load(root_dir + 'maxima_locs.npz')['arr_0']
    maxima_locs = np.random.permutation(maxima_locs)
    
    return list_IDs, voxel_locations, image_dims, maxima_locs, classes


def load_train_test_IDs(list_IDs, predict,
                        exp_type = None,
                        root_dir = None):
    
    if exp_type == "depression_5":
        with open(root_dir + 'meta_data.json', 'r') as f:
            meta_data = json.load(f)
        '''
        meta_data['subj_id']
        meta_data['subj_group']
        meta_data['run_id']
        meta_data['run_type']
        meta_data['music_type']
        '''
        if predict:
            run_ids = np.array(meta_data['run_id'])[list_IDs]
            idxs = np.where((run_ids == 3) + (run_ids == 5))[0]
            list_IDs_part = list_IDs[idxs]
        else:
            run_ids = np.array(meta_data['run_id'])[list_IDs]
            idxs = np.where((run_ids != 3) * (run_ids != 5))[0]
            list_IDs_part = list_IDs[idxs]
    else:
        #choose your desired train/test split
        idxs = np.where(list_IDs >= 0)[0]
        list_IDs_part = list_IDs[idxs]

    return list_IDs_part, idxs