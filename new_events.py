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
data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221114/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221114/proccessed/'

data_name = 'full'
path_data = os.path.join(data_path,data_name +'.fif') 
data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)
events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step') 
filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')
mne.write_events(filename_events,events)