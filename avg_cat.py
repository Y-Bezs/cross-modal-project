# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:38:13 2022

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

data_path =r'Z:/cross_modal_project/Across_participants/Category/'
file_name='scores_'
par=['105','107','108','111','112','115','116','117','118','119','120','121','122','125',
         '126','127','128','129','130','131','132','134','136','137','138','139','140']

#Time_name='scores_time.npy'
Time_name='time_p_natVCman115_sptrl.npy'
path_file = os.path.join(data_path, Time_name)
#time=np.load(path_file)[49:]
time=np.load(path_file)
scr=np.zeros([len(time),len(par)])
a=[]
Category = {
      "11":"move",
      "12":"still",
      "21":"big",
      "22":"small",
      "31":"nat",
      "32":"man"
}
color_v={
       'w':'blue',
       'p':'green',
       }

for mod in ['w','p']:

   for cat in [11,21,31]:
      cat_1=mod+'/'+Category[str(cat)]
      cat_2=mod+'/'+Category[str(cat+1)] 
      
      for sub in range(len(par)):
         filename = op.join(data_path,file_name+ mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+par[sub]+'_sptrl_h0115'+'.npy')
         score=np.load(filename)           
         scr[:,sub]=score
         
      meanWP=np.mean(scr,axis=1)
      stdWP=np.std(scr,axis=1)/np.sqrt(np.size(par))
      filename_all = op.join(data_path,file_name+ mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+'_sptrl_h0115'+'.npy')
      np.save(filename_all, meanWP)
      filename_all = op.join(data_path,file_name+ mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+'_sptrl_h0115'+'.npy')
      np.save(filename_all, stdWP)    

      #time=np.arange(-0.1,0.8,0.9/270)
      y = meanWP
      e = stdWP
      fig=plt.figure()
      #plt.errorbar(time, y, e, linestyle='None', marker='^',color=color_v[mod],markerfacecolor='red')
      plt.plot(time,y,color=color_v[mod])
      plt.axvline(.0, color='k', linestyle='-')
      plt.axvline(.6, color='k', linestyle='-')
      plt.axhline(.5, color='k', linestyle='--', label='chance')
      plt.xlabel('Times')
      plt.ylabel('AUC/scores')
      plt.title(mod+'/ '+Category[str(cat)]+' VC '+Category[str(cat+1)]+'/ supertrials-h_/ '+str(len(par))+'averaged')  # Area Under the Curve
      #plt.show()
      
      fig.savefig(data_path+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+'_head0115.png', dpi=600)
      
      #plt.plot(filtered)
      #plt.ylim=([0.45,0.7])