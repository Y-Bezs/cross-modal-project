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

data_path =r'Z:/cross_modal_project/Across_participants/Category/'
file_name1 = 'scores_all_move_big_nat_W_P'
file_name2='_head_'

par_arr=['105','107','108','111','112','113','114','115','116','117','118','119','120','121']
#par_arr=['105','107','108','111','112','115','116','117','118','119','120','121','122','125',
#         '126','127','128','129','130','131','132','134','136','137','138','139','140']
#delta_T = 15
delta_T = '1_30'
file_name_time = 'times_all_head_' + delta_T+'.npy'
path_file = os.path.join(data_path, file_name_time)
time = np.load(path_file)
scores_all = np.zeros([len(par_arr), 6 , 271])

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

for ii, par in enumerate(par_arr):   
    file_name = file_name1 + par + file_name2 + delta_T +'.npy'
    path_file = os.path.join(data_path, file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
scores_mean = np.mean(scores_all, axis = 0)
scores_std = np.std(scores_all, axis = 0)/np.sqrt(len(par_arr))
  
    
for aa in range(3):
    fig=plt.figure(aa)
    plt.plot(time,scores_mean[aa,:],color=color_v[aa//3], label='Words')
    plt.fill_between(time, scores_mean[aa,:]-scores_std[aa,:], scores_mean[aa,:]+scores_std[aa,:], color=color_v[aa//3], alpha=0.2)

    plt.plot(time,scores_mean[aa+3,:],color=color_v[(aa+3)//3], label='Pictures')
    plt.fill_between(time, scores_mean[aa+3,:]-scores_std[aa+3,:], scores_mean[aa+3,:]+scores_std[aa+3,:], color=color_v[(aa+3)//3], alpha=0.2)
    
    plt.axvline(.0, color='k', linestyle='-')
    plt.axvline(.6, color='k', linestyle='-')
    plt.axhline(.5, color='k', linestyle='--', label='chance')

    plt.xlabel('Times')
    plt.ylabel('AUC/scores') 
    plt.title(Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+'/ '+delta_T+'filter'+' / '+str(len(par_arr))+' averaged') 
    plt.ylim(0.35,0.7) 
    plt.legend(loc='lower right')
    fig.savefig(path_to_save+mod[aa//3]+'_'+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+delta_T+'head.png', dpi=600)
    
    
    
#for aa in range(6):
#    fig=plt.figure(aa)
#    plt.plot(time,scores_mean[aa,:],color=color_v[aa//3])
#
#    plt.fill_between(time, scores_mean[aa,:]-scores_std[aa,:], scores_mean[aa,:]+scores_std[aa,:], color=color_v[aa//3], alpha=0.2)
#    
#    plt.axvline(.0, color='k', linestyle='-')
#    plt.axvline(.6, color='k', linestyle='-')
#    plt.axhline(.5, color='k', linestyle='--', label='chance')
#
#    plt.xlabel('Times')
#    plt.ylabel('AUC/scores') 
#    plt.title(mod[aa//3]+'/ '+Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+'/ '+str(delta_T)+'ms'+' / '+str(len(par_arr))+' averaged') 
#    plt.ylim(0.35,0.7)   
#    fig.savefig(path_to_save+mod[aa//3]+'_'+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+str(delta_T)+'_time.png', dpi=600)
  