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

data_path =r'Z:/cross_modal_project/Across_participants/'
file_name_1='CrossClasssc_CrossClass_p_w.npy'
file_name_2='CrossClasssc_CrossClass_w_p.npy'

path_file = os.path.join(data_path, file_name_1)
CC_p_w=np.load(path_file)

path_file = os.path.join(data_path, file_name_2)
CC_w_p=np.load(path_file)

p_w=[np.mean(CC_p_w,0),np.std(CC_p_w,0)]
w_p=[np.mean(CC_w_p,0),np.std(CC_w_p,0)]
time=np.arange(-0.1,0.8,0.9/271)


y1=p_w[0]-p_w[1]/np.sqrt(14)
y2=p_w[0]+p_w[1]/np.sqrt(14)

fig, axs = plt.subplots(3,sharex=True, sharey=True)
axs = axs.ravel()
clr=['b','r','g']
ttl=['move/still','big/small','nat/mat']
for ii in range(3):
    axs[ii].plot(time,p_w[0][:,ii],color=clr[ii])
    axs[ii].fill_between(time, y1[:,ii], y2[:,ii], color=clr[ii], alpha=.3)
    axs[ii].axhline(.5, color='k', linestyle='--', label='chance')
    axs[ii].axvline(.0, color='k', linestyle='-')
    axs[ii].set_title(ttl[ii])
    print(ii)
fig.suptitle('Train - pic, test - words')

filename_fig = op.join(data_path, 'result_P_W.png')
fig.savefig(filename_fig, dpi=600)

y1=w_p[0]-w_p[1]/np.sqrt(14)
y2=w_p[0]+w_p[1]/np.sqrt(14)

fig, axs = plt.subplots(3,sharex=True, sharey=True)
axs = axs.ravel()
clr=['b','r','g']
ttl=['move/still','big/small','nat/mat']
for ii in range(3):
    axs[ii].plot(time,w_p[0][:,ii],color=clr[ii])
    axs[ii].fill_between(time, y1[:,ii], y2[:,ii], color=clr[ii], alpha=.3)
    axs[ii].axhline(.5, color='k', linestyle='--', label='chance')
    axs[ii].axvline(.0, color='k', linestyle='-')
    axs[ii].set_title(ttl[ii])
    print(ii)
fig.suptitle('Train - words, test - pic')

filename_fig = op.join(data_path, 'result_W_P.png')
fig.savefig(filename_fig, dpi=600)