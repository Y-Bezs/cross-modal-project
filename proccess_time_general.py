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

data_path =r'Y:/cross_modal_project/Across_participants/Time_generalization_wT/'
participant_arr=list(Part_info.keys())[:]

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

sensor ='meg'
time_add = '_50'

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[xx]+'_'+ sensor+'_w_T'+time_add


file_name_time = 'times_' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[xx] + '/matrix/' + file_name_time)
time = np.load(path_file)
scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time), len(time)))

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
    path_file = os.path.join(data_path+suffics[xx] + '/matrix/' + file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
scores_mean = np.mean(scores_all, axis = 0)
scores_std = np.std(scores_all, axis = 0)/np.sqrt(len(participant_arr))
score_mean_std = np.concatenate((scores_mean,scores_std))

score_name_save = '/scores_mean_std_'+suffics[xx]+'_'+sensor+'.npy'
np.save(path_to_save+score_name_save,score_mean_std)
  
    
for aa in range(3):
    fig,ax=plt.subplots()
    im=ax.imshow(score_mean_std[aa,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.45, vmax=0.55)
    ax.set_xticks(range(0,375,25))
    ax.set_xticklabels(time[range(0,375,25)],rotation=45)
    ax.set_yticks(range(0,375,25))
    ax.set_yticklabels(time[range(0,375,25)])        
    ax.set_xlabel('Times Test (ms)' + ' / ' + mod[(aa+3)//3])
    ax.set_ylabel('Times Train (ms)' + ' / ' + mod[aa//3])
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]) 
    plt.colorbar(im)
    fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor+ '_train_' +mod[aa//3], dpi=600)    
    
    fig,ax=plt.subplots()
    im=ax.imshow(score_mean_std[aa+3,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.45, vmax=0.55)
    ax.set_xticks(range(0,375,25))
    ax.set_xticklabels(time[range(0,375,25)],rotation=45)
    ax.set_yticks(range(0,375,25))
    ax.set_yticklabels(time[range(0,375,25)])
    ax.set_xlabel('Times Test (ms)' + ' / ' + mod[aa//3])
    ax.set_ylabel('Times Train (ms)' + ' / ' + mod[(aa+3)//3])
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]) 
    plt.colorbar(im)
    fig.savefig(path_to_save+Category[((aa+3)%3+1)*10+1]+'_VC_'+Category[((aa+3)%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor + '_train_' +mod[(aa+3)//3], dpi=600)    


ind = np.array(range(0,375,5))
sliced_array_shifted = score_mean_std[:, 38:, 38:]
sliced_array = sliced_array_shifted[:,::5,::5]
time_shifted=time+0.024


time_sc=time_shifted[38::25]*1000
time_sc[4]=200

clrst_w=np.load('clst_w.npy')
clrst_p=np.load('clst_p.npy')
 #%%  
for aa in [1]:#range(3):
    fig,ax=plt.subplots()
    im=ax.imshow(sliced_array[aa,:,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.40, vmax=0.6)
    plt.plot((0, 67), (0, 67), 'k:')
    ax.set_xticks(range(0,70,5))
    ax.set_xticklabels(time_sc.astype(int),rotation=45)
    ax.set_yticks(range(0,70,5))
    ax.set_yticklabels(time_sc.astype(int))
    ax.set_xlabel('Times Test (s)' + ' / ' + mod[(aa+3)//3])
    ax.set_ylabel('Times Train (s)' + ' / ' + mod[aa//3])    
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]) 
    plt.colorbar(im)
    plt.contour(clrst_w, 0)
    fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor+ '_train_' +mod[aa//3] + '_sliced', dpi=600)    
    
    fig,ax=plt.subplots()
    im=ax.imshow(sliced_array[aa+3,:,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.40, vmax=0.6)
    plt.plot((0, 67), (0, 67), 'k:')
    ax.set_xticks(range(0,70,5))
    ax.set_xticklabels(time_sc.astype(int),rotation=45)
    ax.set_yticks(range(0,70,5))
    ax.set_yticklabels(time_sc.astype(int))
    ax.set_xlabel('Times Test (ms)' + ' / ' + mod[aa//3])
    ax.set_ylabel('Times Train (ms)' + ' / ' + mod[(aa+3)//3])
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2] ) 
    plt.colorbar(im)
    plt.contour(clrst_p, 0)
    fig.savefig(path_to_save+Category[((aa+3)%3+1)*10+1]+'_VC_'+Category[((aa+3)%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor + '_train_' +mod[(aa+3)//3]+ '_sliced', dpi=600)    
 
sliced_array_w=sliced_array[1,:,:]    
sliced_array_p=sliced_array[4,:,:]   
 
# %%
xx=2

data_path =r'Y:/cross_modal_project/Across_participants/Time_generalization/'
participant_arr=list(Part_info.keys())[:]

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

sensor ='meg'
time_add = '_50'

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[xx]+'_'+ sensor


file_name_time = 'times_' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[xx] + '/matrix/' + file_name_time)
time = np.load(path_file)
scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time), len(time)))

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
    path_file = os.path.join(data_path+suffics[xx] + '/matrix/' + file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
scores_mean = np.mean(scores_all, axis = 0)
scores_std = np.std(scores_all, axis = 0)/np.sqrt(len(participant_arr))
score_mean_std = np.concatenate((scores_mean,scores_std))

score_name_save = '/scores_mean_std_'+suffics[xx]+'_'+sensor+'.npy'
np.save(path_to_save+score_name_save,score_mean_std)
  
    
for aa in range(3):
    fig,ax=plt.subplots()
    im=ax.imshow(score_mean_std[aa,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.45, vmax=0.55)
    ax.set_xticks(range(0,375,25))
    ax.set_xticklabels(time[range(0,375,25)],rotation=45)
    ax.set_yticks(range(0,375,25))
    ax.set_yticklabels(time[range(0,375,25)])        
    ax.set_xlabel('Times Test (ms)' + ' / ' + mod[(aa+3)//3])
    ax.set_ylabel('Times Train (ms)' + ' / ' + mod[aa//3])
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]) 
    plt.colorbar(im)
    fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor+ '_train_' +mod[aa//3], dpi=600)    
    
    fig,ax=plt.subplots()
    im=ax.imshow(score_mean_std[aa+3,:], interpolation='nearest', origin='lower', cmap='RdBu_r',
                 vmin=0.45, vmax=0.55)
    ax.set_xticks(range(0,375,25))
    ax.set_xticklabels(time[range(0,375,25)],rotation=45)
    ax.set_yticks(range(0,375,25))
    ax.set_yticklabels(time[range(0,375,25)])
    ax.set_xlabel('Times Test (ms)' + ' / ' + mod[aa//3])
    ax.set_ylabel('Times Train (ms)' + ' / ' + mod[(aa+3)//3])
    ax.set_title( Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]) 
    plt.colorbar(im)
    fig.savefig(path_to_save+Category[((aa+3)%3+1)*10+1]+'_VC_'+Category[((aa+3)%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor + '_train_' +mod[(aa+3)//3], dpi=600)    



score_all_w=np.mean(scores_all[:,0:3,:,:],axis=1)
score_all_p=np.mean(scores_all[:,3:6,:,:],axis=1)
