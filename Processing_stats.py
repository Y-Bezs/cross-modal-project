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

data_path =r'Y:/cross_modal_project/Cross_Class/max_wo_head/'
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

aa=1 #big vc small
up_down_ppt=np.zeros(( len(participant_arr) , len(['w', 'p']) * len([11, 21, 31]) ,2 ))
for ii, par_name in enumerate(participant_arr):
    par=str(Part_info[par_name])
    file_name = file_name1 + par + file_name2 +'.npy'
    path_file = os.path.join(data_path+suffics[xx] + '/matrix/' + file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
    scores_flt=np.fliplr(scores[:,60:,60:])
    total_sum_up = np.sum(np.triu(scores_flt, k=1),axis=(1,2))  # extract upper triangle and sum off-diagonal elements  
    total_sum_down = np.sum(np.tril(scores_flt, k=-1),axis=(1,2))  # extract upper triangle and sum off-diagonal elements
    
    up_down_ppt[ii,:,0] = total_sum_up
    up_down_ppt[ii,:,1] = total_sum_down

from scipy import stats
stats.ttest_rel(up_down_ppt[:,1,0],up_down_ppt[:,1,1])
    
    plt.imshow(A)
    plt.colorbar(up_down_ppt[:,1,0],up_down_ppt[:,1,1])
    
score_size_train_w=scores_all[:,1,:]
score_size_train_p=scores_all[:,4,:]    

#%%
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
participant_arr=list(Part_info.keys())[:]

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

sensor ='meg'
time_add = '_50'

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[xx]+'_'+ sensor+time_add
file_name_time = 'times_' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[xx] +'/'+ file_name_time)
time = np.load(path_file)
scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))

start_ind=175
end_ind=300
for ii, par_name in enumerate(participant_arr):
    par=str(Part_info[par_name])
    file_name = file_name1 + par + file_name2 +'.npy'
    path_file = os.path.join(data_path+'/'+suffics[xx], file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores
    
total_sum = np.mean(scores_all[:,:,start_ind:end_ind],axis=2)
from scipy import stats
stats.ttest_1samp(total_sum[:,1],0.5,axis=0)   
    












    
    