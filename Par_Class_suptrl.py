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
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'

data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221129/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221129/proccessed/'
old=0

data_name = 'full'
path_file = os.path.join(result_path,data_name+'_supertrials.fif') 
epochs = mne.read_epochs(path_file, preload=True,verbose=True)

#if old==1:
#        filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')
#else:
#        filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')
#events = mne.read_events(filename_events, verbose=True)

events_id = {'start_000/w/still/small/man':240+1,'start_100/w/move/small/man':32+1, 'start_010/w/still/big/man':64+1, 'start_110/w/move/big/man':96+1, 
        'start_001/w/still/small/nat':128+1, 'start_101/w/move/small/nat':160+1, 'start_011/w/still/big/nat':192+1, 'start_111/w/move/big/nat':224+1,
        'start_000/p/still/small/man':240+2,'start_100/p/move/small/man':32+2, 'start_010/p/still/big/man':64+2, 'start_110/p/move/big/man':96+2,
        'start_001/p/still/small/nat':128+2, 'start_101/p/move/small/nat':160+2, 'start_011/p/still/big/nat':192+2, 'start_111/p/move/big/nat':224+2}


#pic=mne.event.match_event_names(event_names=events_id,keys=['p'])
#word=mne.event.match_event_names(event_names=events_id,keys=['w'])

epochs_1 = epochs['p'].copy()
epochs_2 = epochs['w'].copy()
epochs_rs_raw=mne.concatenate_epochs([epochs_1,epochs_2])
epochs_rs=epochs_rs_raw.copy().filter(0,30)
epochs_rs.resample(300)
#epochs_rs.crop(tmin=-0.1, tmax=0.8)

tr_1=np.unique(epochs_1.events[:,2])
tr_2=np.unique(epochs_2.events[:,2])

events_n=epochs_rs_raw.events
merged_events = mne.merge_events(events_n, tr_1, 1)
merged_events = mne.merge_events(merged_events, tr_2, 2)

epochs_rs.events=merged_events

X = epochs_rs.get_data(picks='meg') 
X.shape
y = merged_events[:,2]
print(y)

clf = make_pipeline(Vectorizer(),StandardScaler(),  
                   LinearModel(sklearn.svm.SVC(kernel = 'linear'))) 

time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)   
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
scores = np.mean(scores, axis=0)
np.save(result_path+'scores_sptrl_wp', scores)
np.save(result_path+'scores_time_sptrl', epochs_rs.times)
#fig, ax = plt.subplots()
#plt.ylim([0.35, 0.65])
fig=plt.figure()
plt.plot(epochs_rs.times, scores, label='score')
#plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC')  # Area Under the Curve
plt.legend()
plt.axvline(.0, color='k', linestyle='-')
plt.title('Words vs Pictures/ supertr')
filename_fig = op.join(result_path, 'WordPic_sptr.png')
fig.savefig(filename_fig, dpi=600)
