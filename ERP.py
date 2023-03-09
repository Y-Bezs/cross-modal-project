# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:36:02 2022

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

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'
data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221100/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221100/proccessed/'

data_name = 'full'
path_data = os.path.join(data_path,data_name +'.fif') 
data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

#%%  MARK BREAKS
events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step') 

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

annotations_break=mne.preprocessing.annotate_break(data_raw, events=events,
                                             min_break_duration=7, 
                                             t_start_after_previous=0.1, 
                                             t_stop_before_next=0.1, 
                                             ignore=('bad', 'edge'))
data_raw.set_annotations(data_raw.annotations+annotations_break)  

#%%
#file_sss_path =r'Z:/cross_modal_project/'
file_sss_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'
crosstalk_file = os.path.join(file_sss_path,'ct_sparse_SA.fif')
cal_file = os.path.join(file_sss_path,'sss_cal_SA.dat')

#fig1=dataW1.plot_psd(fmax=60);
#data_raw.plot(duration=5,n_channels=10,scalings='auto')

data_raw.info['bads'] = []
data_raw_check = data_raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
    data_raw_check, 
    cross_talk=crosstalk_file, 
    skip_by_annotation=('edge', 'bad_acq_skip','BAD'),
    calibration=cal_file,
    return_scores=True, 
    verbose=True)

print('noisy =', auto_noisy_chs)
print('flat = ', auto_flat_chs)

data_raw.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
#%%
data_raw.fix_mag_coil_types()

data_raw_sss = mne.preprocessing.maxwell_filter(
    data_raw,
    cross_talk=crosstalk_file,
    calibration=cal_file,
    verbose=True)
fig1=data_raw.plot_psd(fmax=60, n_fft = 1000);
fig2=data_raw_sss.plot_psd(fmax=60, n_fft = 1000);
filename_fig = op.join(result_path,'PSD_before.png')
fig1.savefig(filename_fig, dpi=600)
filename_fig = op.join(result_path,file_name[0] + 'filtered.png')
fig2.savefig(filename_fig, dpi=600)

#%%

path_file_results = os.path.join(result_path,data_name+'sss-1.fif') 
path_file_results
data_raw_sss.save(path_file_results,overwrite=True) 

#%%

eog_events = mne.preprocessing.find_eog_events(data_raw, ch_name='EOG002',thresh=0.0002) 
n_blinks = len(eog_events)  
onset = eog_events[:, 0] / data_raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
description = ['blink'] * n_blinks
orig_time = data_raw.info['meas_date']
annotations_blink = mne.Annotations(onset,duration,description,orig_time)

threshold_muscle = 10  
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    data_raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])

data_raw.set_annotations(annotations_blink+annotations_muscle+data_raw.annotations)
path_file_results = os.path.join(result_path,data_name + 'data_blink.fif') 
data_raw.save(path_file_results,overwrite=True) 

path_file_results = os.path.join(result_path,data_name +'ann-1.fif') 
data_raw.save(path_file_results,overwrite=True) 
#%%plot eog

#eog_epochs=mne.preprocessing.create_eog_epochs(data_raw,ch_name='EOG002',thresh=0.0002)
#eog_epochs_a=eog_epochs.average()
#eog_epochs_a.plot_joint()

#%% ICA

data_raw_resmpl = data_raw.copy().pick_types(meg=True)
data_raw_resmpl.resample(200)
data_raw_resmpl.filter(1, 40)

ica = ICA(method='fastica',
    random_state=97,
    n_components=30,
    verbose=True)

ica.fit(data_raw_resmpl,
    verbose=True)

ica.exclude = [0, 3, 8, 23]
path_file = os.path.join(result_path,data_name + 'ann-1' + '.fif') 
path_outfile = os.path.join(result_path,data_name +'ica-1' + '.fif') 
raw_ica = mne.io.read_raw_fif(path_file,allow_maxshield=True,verbose=True,preload=True)   
ica.apply(raw_ica)
raw_ica.save(path_outfile,overwrite=True) 


data_raw=raw_ica

#%% EVENTS

#events = mne.find_events(data_raw, stim_channel='STI101',min_duration=0.001001)
events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)))                

filename_events = op.join(result_path,data_name + 'eve-all' +'.fif')
mne.write_events(filename_events,events,overwrite=True)

cat=np.array([240, 128, 64, 32, 192, 160, 96, 224])
onset_tr_wrd=cat+1
onset_tr_pic=cat+2

raw_list = list()
events_list = list()

path_file = os.path.join(result_path,data_name+'ica-1' + '.fif') 
raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True)

    
events = mne.read_events(filename_events, verbose=True)

if np.where(events[:,2]==24)[0].size==0:
    events_id = {'fix':255, 'break':1,'qst_left':5,'qst_right':6,'ans_left':8,'ans_right':16,
        'start_000/w':240+1,'start_100/w':32+1, 'start_010/w':64+1, 'start_110/w':96+1, 'start_001/w':128+1, 'start_101/w':160+1, 'start_011/w':192+1, 'start_111/w':224+1,
        'start_000/p':240+2,'start_100/p':32+2, 'start_010/p':64+2, 'start_110/p':96+2, 'start_001/p':128+2, 'start_101/p':160+2, 'start_011/p':192+2, 'start_111/p':224+2,
        'words':191,'pictures':127}
else:
        events_id = {'fix':255, 'break':1,'qst_left':5,'qst_right':6,'ans_left':8,'ans_right':16,
        'start_000/w':240+1,'start_100/w':32+1, 'start_010/w':64+1, 'start_110/w':96+1, 'start_001/w':128+1, 'start_101/w':160+1, 'start_011/w':192+1, 'start_111/w':224+1,
        'start_000/p':240+2,'start_100/p':32+2, 'start_010/p':64+2, 'start_110/p':96+2, 'start_001/p':128+2, 'start_101/p':160+2, 'start_011/p':192+2, 'start_111/p':224+2,
        'words':191,'pictures':127}
raw_list.append(raw)
events_list.append(events)

#merged_words = mne.merge_events(events, cat+1, 100)
#merged_pic = mne.merge_events(events, cat+2, 200)

raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
del raw_list

epochs = mne.Epochs(raw,
            events, events_id,
            tmin=-0.40 , tmax=1,
            baseline=None,
            proj=True,
            picks = 'all',
            detrend = 1,
            #reject=reject,
            #reject_by_annotation=True,
            preload=True,
            verbose=True)

path_outfile = os.path.join(result_path,data_name+'_epo.fif') 
epochs.save(path_outfile,overwrite=True)

match_event_names(
    event_names=event_names,
    keys=['auditory', 'left']
)

epochs['start_000_w','start_001_w','start_010_w','start_100_w','start_101_w','start_011_w','start_111_w','start_110_w'].filter(0.0,30).crop(-0.1,0.4).plot_image(picks=['MEG2343'],vmin=-150,vmax=150);

epochs['start_000_p','start_001_p','start_010_p','start_100_p','start_101_p','start_011_p','start_111_p','start_110_p'].filter(0.0,30).crop(-0.1,0.4).plot_image(picks=['MEG2343'],vmin=-150,vmax=150);


filename_fig = op.join(result_path, 'ERP_grad.png')
savefig(filename_fig, dpi=600)