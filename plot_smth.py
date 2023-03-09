# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:24:54 2022

@author: yxb968
"""


import os 
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as op

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


result_path=r'Z:/cross_modal_project/Across_participants/Word_Pic/'
file_name='scores_avg_all'
file_name_avg='scores_avg_sptr'
err='err_'

path_file = os.path.join(result_path,file_name + '.npy')
score_all=np.load(path_file)

path_file = os.path.join(result_path,file_name_avg + '.npy')
score_avg=np.load(path_file)

path_file = os.path.join(result_path,'scores_time.npy')
time_scr=np.load(path_file)

path_file = os.path.join(result_path,err+file_name + '.npy')
err_all=np.load(path_file)

path_file = os.path.join(result_path,err+file_name_avg + '.npy')
err_avg=np.load(path_file)


result_path=r'Z:/cross_modal_project/Across_participants/Word_Pic/'
path_file = os.path.join(result_path, 'scores_time.npy')
time=np.load(path_file)


y1 = score_all
y2 = score_avg
e1 = err_all
e2 = err_avg
fig=plt.figure()
plt.errorbar(time, y1, e1, linestyle='None', marker='^',color='black',markerfacecolor='green',label='all trials')
plt.errorbar(time, y2, e2, linestyle='None', marker='o',color='blue',markerfacecolor='red',label='supertrials')
#plt.plot(time, y1,color='black')
#plt.plot(time_scr, y2,color='blue')
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.legend(loc=' lower center')
plt.title('Words VC Pictures (averaged 9 subjects)')
plt.xlabel('time, s')
plt.ylabel('AUC')

plt.show()
filename_fig = op.join(result_path, 'all_trials_VC_supertrials_WP.png')
fig.savefig(filename_fig, dpi=600)
