

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_result(dmfa, classes,
                fig_PATH, prefix = '',
                ext = ".pdf", data_st = None,
                days = None, predict = True, ID = None):
    if ID is not None:
        ID_name = np.array(['guangzhou','birmingham','hangzhou','seattle'])
        data_id = np.where(ID_name == ID)[0][0]
    
    n_class = dmfa.p_c.size(-1)
    T = dmfa.q_w_mu.size(1)
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    z_0_p = dmfa.z_0_mu.detach().numpy()
    z_0_p_sig = dmfa.z_0_sig.exp().detach().numpy()
    fig = plt.figure()
    colors = ['b','r','g','y']
    labels = ['group%d'%(c+1) for c in range(n_class)]
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
    
    dataa, data_mean, data_std = data_st
    dataa = dataa.reshape(-1,dataa.shape[-1])*data_std+data_mean

    ws = zss.reshape(-1,dmfa.q_w_mu.shape[-1])
    z_t_1 = dmfa.q_z_0_mu[-1].reshape(1, -1)
    for i in range(T):
        u_t_1 = torch.zeros(1, 0)
        p_w_mu, p_w_sig = dmfa.temp(z_t_1)
        p_z_mu, p_z_sig = dmfa.trans(z_t_1, u_t_1)
        ws = np.concatenate((ws, z_t_1.detach().numpy()), axis = 0)
        z_t_1 = p_z_mu * 1.0
    f_locs = dmfa.q_F_loc_mu.detach().numpy()
    y_pred = np.matmul(ws, f_locs)*data_std+data_mean

    if predict:    
        if ID is not None:
            titles = ['Guangzhou road segment',
                      'Birmingham car park',
                      'Hangzhou metro staion',
                      'Seattle loop detector']
            ylabels = ['Traffic speed',
                       'Occupancy',
                       'Passenger flow',
                       'Traffic speed']
            ticks = [np.arange(144/2, 144*2, 144),
                     np.arange(18/2, 18*2, 18),
                     np.arange(108/2, 108*2, 108),
                     np.arange(288/2, 288*2, 288)]
            ticklabels = [['Sep. 30', 'Oct. 01'],
                          ['Dec. 19','Dec. 20'],
                          ['Jan. 25','Jan. 26'],
                          ['Jan. 28', 'Jan. 29']]
    
        fontsize= 12

        for idx_loc in range(0, dmfa.q_F_loc_mu.shape[-1], dmfa.q_F_loc_mu.shape[-1]//5):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(dataa[-T:,idx_loc], label = "Actual")
            ax.plot(y_pred[-2*T:-T,idx_loc], 'r-',label = "Recovered", alpha = 0.8)
            y_preds = y_pred[-2*T:,idx_loc] * 1.0
            y_preds[:T] = np.nan
            ax.plot(y_preds, 'r-.', label = "Predicted", alpha = 0.8)
            ax.legend(framealpha = 0, fontsize=13)
            if ID is not None:
                ax.set_title(titles[data_id]+' #%d'%idx_loc, fontsize=fontsize)
                ax.set_ylabel(ylabels[data_id], fontsize=fontsize+2)
                ax.set_ylim(ymin=0)
                ax.set_xticks(ticks[data_id])
                ax.set_xticklabels(ticklabels[data_id], fontsize=fontsize)
            fig.savefig(fig_PATH + "%sprediction_long_term%s" %(prefix,idx_loc) + ext, bbox_inches='tight')
        plt.close('all')

        idxs = np.where(dataa[-days*T:] != 0)
        RMSE = np.sqrt(np.power(dataa[-days*T:][idxs] - y_pred[-(days+1)*T:-T][idxs],2).mean())
        print('Test RMSE %.2f' %RMSE)
        MAPE = (np.absolute(dataa[-days*T:][idxs] - y_pred[-(days+1)*T:-T][idxs])/dataa[-days*T:][idxs]).mean()*100
        print('Test MAPE %.2f' %MAPE)
    else:
        idxs = np.where(dataa[:-days*T] != 0)
        RMSE = np.sqrt(np.power(dataa[:-days*T][idxs] - y_pred[:-(days+1)*T][idxs],2).mean())
        print('Train RMSE %.2f' %RMSE)
        MAPE = (np.absolute(dataa[:-days*T][idxs] - y_pred[:-(days+1)*T][idxs])/dataa[:-days*T][idxs]).mean()*100
        print('Train MAPE %.2f' %MAPE)
        
    if predict:        
        ws = zss.reshape(-1,dmfa.q_w_mu.shape[-1])
        for j in range(days, 0 , -1):
            z_t_1 = dmfa.q_z_0_mu[-j].reshape(1, -1)
            p_w_mu, p_w_sig = dmfa.temp(z_t_1)
            ws = np.concatenate((ws, z_t_1.detach().numpy()), axis = 0)
            for i in range(0, T-1):
                u_t_1 = torch.zeros(1, 0)
                p_z_mu, p_z_sig = dmfa.trans(z_t_1, u_t_1)
                p_w_mu, p_w_sig = dmfa.temp(p_z_mu)
                z_t_1 = dmfa.q_z_mu[-j,i].reshape(1,-1)
                ws = np.concatenate((ws, p_z_mu.detach().numpy()), axis = 0)
        f_locs = dmfa.q_F_loc_mu.detach().numpy()
        y_pred = np.matmul(ws, f_locs)*data_std+data_mean
        
        if ID is not None:
            ticks = [np.arange(144/2, 144*days, 144),
                     np.arange(18/2, 18*days, 18),
                     np.arange(108/2, 108*days, 108),
                     np.arange(288/2, 288*days, 288)]
            ticklabels = [['Sep. 26', 'Sep. 27', 'Sep. 28', 'Sep. 29', 'Sep. 30'],
                          ['Dec. 13', 'Dec. 14', 'Dec. 15','Dec. 16','Dec. 17','Dec. 18','Dec. 19'],
                          ['Jan. 21', 'Jan. 22', 'Jan. 23','Jan. 24','Jan. 25'],
                          ['Jan. 24', 'Jan. 25', 'Jan. 26', 'Jan. 27', 'Jan. 28']]

        for idx_loc in range(0, dmfa.q_F_loc_mu.shape[-1], dmfa.q_F_loc_mu.shape[-1]//5):
            fig = plt.figure(figsize=(7,3))
            ax = fig.add_subplot(111)
            ax.plot(dataa[-T*days:,idx_loc], label = "Actual")
            y_preds = y_pred[-T*days:,idx_loc]
            ax.plot(y_preds, 'r-', label = "Predicted", alpha = 0.8)
            ax.legend(framealpha = 0, fontsize=13)
            if ID is not None:
                ax.set_title(titles[data_id]+' #%d'%idx_loc, fontsize=fontsize)
                ax.set_ylabel(ylabels[data_id], fontsize=fontsize+2)
                ax.set_xticks(ticks[data_id])
                ax.set_xticklabels(ticklabels[data_id], fontsize=fontsize)
            fig.savefig(fig_PATH + "%sprediction_roll_short%s" %(prefix,idx_loc) + ext, bbox_inches='tight')
    
        RMSE = np.sqrt(np.power(dataa[-days*T:][idxs] - y_pred[-days*T:][idxs],2).mean())
        print('Prediction RMSE %.2f' %RMSE)
        MAPE = (np.absolute(dataa[-days*T:][idxs] - y_pred[-days*T:][idxs])/dataa[-days*T:][idxs]).mean()*100
        print('Prediction MAPE %.2f' %MAPE)
    
        plt.close('all')