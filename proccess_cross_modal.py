# -*- coding: utf-8 -*-
import os
import numpy as np
import mne
import matplotlib
import sys
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mne.preprocessing import annotate_muscle_zscore
import os.path as op
from mne.preprocessing import ICA
import scipy.io
import seaborn as sns
from init_y import *

xx=2

data_path =r'Y:/cross_modal_project/Across_participants/Cross_Class/'
participant_arr=list(Part_info.keys())
suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

sensor ='meg'
time_add = '_50'
file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[xx]+'_'+ sensor + time_add


file_name_time = 'times_' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[xx] +'/'+ file_name_time)
time = np.load(path_file)
scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))

path_to_save = data_path + 'results/' 
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

color_v={
       0:'blue',
       1:'green',
       }

mod={
       0:'WORDS',
       1:'PICTURES',
       }

Category = {
      11:"move",
      12:"still",
      21:"big",
      22:"small",
      31:"nat",
      32:"man"
}

for ii, par_name in enumerate(participant_arr):
    par=str(Part_info[par_name])
    file_name = file_name1 + par + file_name2 +'.npy'
    path_file = os.path.join(data_path+'/'+suffics[xx], file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
scores_mean = np.mean(scores_all, axis = 0)
scores_std = np.std(scores_all, axis = 0)/np.sqrt(len(participant_arr))
score_mean_std = np.concatenate((scores_mean,scores_std))

for aa in range(3):
    fig=plt.figure(aa)
    #plt.plot(time,scores_mean[aa,:],color=color_v[aa//3], label='train_Words')
    #plt.fill_between(time, scores_mean[aa,:]-scores_std[aa,:], scores_mean[aa,:]+scores_std[aa,:], color=color_v[aa//3], alpha=0.2)

    plt.plot(time,scores_mean[aa+3,:],color=color_v[(aa+3)//3], label='train_Pictures')
    #plt.fill_between(time, scores_mean[aa+3,:]-scores_std[aa+3,:], scores_mean[aa+3,:]+scores_std[aa+3,:], color=color_v[(aa+3)//3], alpha=0.2)
    
    plt.axvline(.0, color='k', linestyle='-')
    plt.axvline(.6, color='k', linestyle='-')
    plt.axhline(.5, color='k', linestyle='--', label='chance')

    plt.xlabel('Times')
    plt.ylabel('AUC/scores') 
    plt.title(Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+'/ '+suffics[xx]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
    plt.ylim(0.35,0.7) 
    plt.legend(loc='lower right')
    fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor + time_add+'p' , dpi=600)
# %%





