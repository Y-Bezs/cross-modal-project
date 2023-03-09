# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:37:53 2022

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

#how to save events:
events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step') 
filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')
mne.write_events(filename_events, events,verbose=True,overwrite=True)

#my code to chande events.
data_path =r'Z:/cross_modal_project/221118/'
result_path=r'Z:/cross_modal_project/221118/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'
#data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221110/'
#result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221110/proccessed/'

data_name = 'full'
path_data = os.path.join(data_path,data_name +'.fif') 
#data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')

#%%  MARK BREAKS
events = mne.read_events(filename_events, verbose=True)

cat=np.array([240, 128, 64, 32, 192, 160, 96, 224])

#num1 = [len(np.where(events[:,2]==x)[0]) for x in cat+2]
#duration=np.empty((0))
#onset=np.empty((0))
#description=[]
#breaks_tmp=np.where(events[:,2]==1)[0]
#for i in range(0,len(breaks_tmp)-1):
#    duration=np.append(duration,events[breaks_tmp[i]+2,0]-events[breaks_tmp[i],0])
#    onset=np.append(onset,events[breaks_tmp[i],0])
#i=i+1
#duration=np.append(duration,events[breaks_tmp[i]+1,0]-events[breaks_tmp[i],0])
#onset=np.append(onset,events[breaks_tmp[i],0])
#num_breaks=len(breaks_tmp)
#orig_time = data_raw.info['meas_date']
#annototations_break = mne.Annotations(onset=onset/1000,  # in seconds
#                           duration=duration/1000,  # in seconds, too
#                           description=repeat('BAD', num_breaks),orig_time=orig_time)

#data_raw.set_annotations(annototations_break) 



ind=scipy.io.loadmat(data_path+'index.mat')
new_trig=ind['index'][np.where(ind['index']!=-1)[0]]

events_up=events
tmp_ind=np.where((events[:,2]==cat[0]) | (events[:,2]==cat[1])  | (events[:,2]==cat[3])  | (events[:,2]==cat[4])
                  | (events[:,2]==cat[5])  | (events[:,2]==cat[6])  | (events[:,2]==cat[7])  | (events[:,2]==cat[2]))

events_up[tmp_ind[0],2]=new_trig.T

filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')
mne.write_events(filename_events,events_up,overwrite=True)
