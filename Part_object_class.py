import os

import numpy as np
import scipy
from sklearn.discriminant_analysis import _cov
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from cv import ShuffleBinLeaveOneOut

np.random.seed(10)
sub=1

participant_arr=list(Part_info.keys())
participant=participant_arr[sub-1]
data_path  ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
result_path=data_path+'proccessed/'
old=1 if Part_info[participant]<109 else 0

filename_events = op.join(result_path,data_name + '_eve-right' +'.fif')
events = mne.read_events(filename_events, verbose=True)

events_arr = list(events_id.keys())
triggers = np.array([events_id[eve] for eve in events_arr])
events_new = events.copy()
for a, tr in enumerate(triggers[0:9]):
    mask_w = events_new[:, 2] == tr
    events_new[mask_w, 2] = 210 + a*10
    mask_p = events_new[:, 2] == tr+1
    events_new[mask_p, 2] = 110 + a*10

for ii in [41,42,43,44,45,46]:
    if old==1:
        shift = 2 if participant=='221114' else 1
    else:
        shift=2
    events_new[np.where(events[:,2]==ii)[0]-shift,:] +=ii-41


# Load data and trial labels for the two sessions of participant 01
sessions = [
    # Session 1
    dict(
        data=np.load(os.path.join(root, 'data01_sess1.npy')),
        # data has shape n_trials x n_sensors x n_timepoints
        labels=np.load(os.path.join(root, 'labels01_sess1.npy'))
        # labels has shape 1 x n_trials (i.e., one condition label [object category] per trial)
    ),
    # Session 2
    dict(
        data=np.load(os.path.join(root, 'data01_sess2.npy')),
        labels=np.load(os.path.join(root, 'labels01_sess2.npy'))
    )
]