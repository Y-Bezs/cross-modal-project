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

data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221124/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221124/proccessed/'
old=0

data_name = 'full'
path_file = os.path.join(result_path,data_name+'_epo_right.fif') 
epochs = mne.read_epochs(path_file, preload=True,verbose=True)

#if old==1:
#        filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')
#else:
filename_events = op.join(result_path,data_name + '_eve-right' +'.fif')
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

#pic=mne.event.match_event_names(event_names=events_id,keys=['p'])
#word=mne.event.match_event_names(event_names=events_id,keys=['w'])

epochs_rs_w = epochs['w'].copy()
epochs_rs_p = epochs['p'].copy()
epochs_rs_raw=mne.concatenate_epochs([epochs_rs_w ,epochs_rs_p])
epochs_rs=epochs_rs_raw.copy().filter(0,30)
epochs_rs.resample(100)
#epochs_rs.crop(tmin=-0.1, tmax=0.8)

cat=np.array([240, 128, 64, 32, 192, 160, 96, 224])
onset_tr_wrd=cat+1
onset_tr_pic=cat+2

events_n=epochs_rs_raw.events
merged_events = mne.merge_events(events_n, onset_tr_wrd, 1)
merged_events = mne.merge_events(merged_events, onset_tr_pic, 2)

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
np.save(result_path+'scores', scores)
np.save(result_path+'scores_time', epochs_rs.times)
#fig, ax = plt.subplots()
#plt.ylim([0.35, 0.65])
fig=plot(epochs_rs.times, scores, label='score')
#plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC')  # Area Under the Curve
plt.legend()
plt.axvline(.0, color='k', linestyle='-')
plt.title('Words vs Pictures')
filename_fig = op.join(result_path, 'WordPic.png')
fig.savefig(filename_fig, dpi=600)
