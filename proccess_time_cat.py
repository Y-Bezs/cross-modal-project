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

data_path =r'Y:/cross_modal_project/Across_participants/Category_w_time/'
participant_arr=list(Part_info.keys())

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

sensor ='mag'

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[xx]+'_'+ sensor+'_50'


file_name_time = 'times' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[xx], file_name_time)
time = np.load(path_file)
scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))

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
    
    
start_ind = np.where(time==0.430)
end_ind = np.where(time==0.500)
total_sum = np.mean(scores_all[:,:,int(start_ind[0]):int(end_ind[0])],axis=2)
from scipy import stats
stats.ttest_1samp(total_sum[:,4],0.5,axis=0)   
    

scores_mean = np.mean(scores_all, axis = 0)
scores_std = np.std(scores_all, axis = 0)/np.sqrt(len(participant_arr))
score_mean_std = np.concatenate((scores_mean,scores_std))

score_name_save = '/scores_mean_std_'+suffics[xx]+'_'+sensor+'.npy'
np.save(path_to_save+score_name_save,score_mean_std)
  
    
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
    plt.title(Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+'/ '+suffics[xx]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
    plt.ylim(0.35,0.7) 
    plt.legend(loc='lower right')
    fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor, dpi=600)
# %%
aa=0
sensor='grad' 
color_x={1:'m-',
         2:'b-',
         3:'k-'}
clr={1:'m',2:'b',3:'k'}
fig=plt.figure()   
for xxx in [1,2,3]:
    score_name = 'scores_mean_std_'+suffics[xxx]+'_'+sensor+'.npy' 
    path_file = os.path.join(path_to_save, score_name)
    scores_mean = np.load(path_file)    
    plt.plot(time,scores_mean[aa+3,:],color_x[xxx], label=suffics[xxx])
    #plt.fill_between(time, scores_mean[aa+3,:]-scores_mean[aa+9,:], scores_mean[aa+3,:]+scores_mean[aa+9,:], color=clr[xxx], alpha=0.2)
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC/scores')
plt.title('Pic_'+Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
plt.ylim(0.35,0.7)
plt.legend(loc='lower right') 
fig.savefig(path_to_save+'Pic_'+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+sensor+'_lines', dpi=600)
    
# %%
aa=0
xxx=1
color_x={0:'k-',
         1:'r-',
         2:'g-'}
clr={0:'k',1:'r',2:'g'}
fig=plt.figure()   
for ii,sensor in enumerate(['meg','grad','mag']):
    score_name = 'scores_mean_std_'+suffics[xxx]+'_'+sensor+'.npy' 
    path_file = os.path.join(path_to_save, score_name)
    scores_mean = np.load(path_file)    
    plt.plot(time,scores_mean[aa,:],color_x[ii], label=sensor)
    #plt.fill_between(time, scores_mean[aa,:]-scores_mean[aa+9,:], scores_mean[aa,:]+scores_mean[aa+6,:], color=clr[ii], alpha=0.2)
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC/scores')
plt.title('Word_'+Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+' / '+str(len(participant_arr))+' averaged / '+suffics[xxx]) 
plt.ylim(0.35,0.7)
plt.legend(loc='lower right') 
fig.savefig(path_to_save+'Word_'+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xxx], dpi=600)
    
        
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
  