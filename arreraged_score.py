# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:26:08 2022

@author: yxb968
"""


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

data_path =r'Z:/cross_modal_project/Across_participants/Word_Pic/'
file_name='scores_W_vc_P'
par=['105','107','108','111','112','113','114','115','116']
Time_name='scores_time.npy'
path_file = os.path.join(data_path, Time_name)
time=np.load(path_file)
scr=np.zeros([time.shape[0],len(par)])
a=[]

for subfile in range(len(par)):
    path_file = os.path.join(data_path,file_name + par[subfile] + '_all.npy')
    score_all=np.load(path_file)
    a.append(len(score_all))
    scr[:,subfile]=score_all
    
meanWP=np.mean(scr,axis=1)
stdWP=np.std(scr,axis=1)/np.sqrt(np.size(par))

 
y = meanWP
e = stdWP
np.save(data_path+'scores_avg_all', y)
np.save(data_path+'err_scores_avg_all', e)
        
fig=plt.figure()
plt.errorbar(time, y, e, linestyle='None', marker='^',color='black',markerfacecolor='red')
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC/scores')
plt.title('Words VC Pictures / all trials (averaged 9)')  # Area Under the Curve
plt.show()
filename_fig = op.join(data_path, 'all_tr_9.png')
fig.savefig(filename_fig, dpi=600)