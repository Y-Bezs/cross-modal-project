
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
data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221024_s/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221024_s/proccessed/'

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
#dataW1.plot(duration=5,n_channels=58,scalings='auto')

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
filename_fig = op.join(result_path,'PSD_filtered.png')
fig2.savefig(filename_fig, dpi=600)

#%%

path_file_results = os.path.join(result_path,data_name+'-sss-1.fif') 
path_file_results
data_raw_sss.save(path_file_results,overwrite=True) 

#%%
#,thresh=0.0002
eog_events = mne.preprocessing.find_eog_events(data_raw, ch_name='EOG002') 
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
path_file_results = os.path.join(result_path,data_name + '-data_blink.fif') 
data_raw.save(path_file_results,overwrite=True) 

path_file_results = os.path.join(result_path,data_name +'-ann-1.fif') 
data_raw.save(path_file_results,overwrite=True) 