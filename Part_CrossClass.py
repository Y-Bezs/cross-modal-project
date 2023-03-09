from init_y import *
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score

result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/CrossClass'

aa = 4
sub = 1
Part_info={
        '221107':105,
        '221110':107,
        '221114':108,
        '221121':111,
        '221124':112,
        '221125':113,
        '221128':114,
        '221129':115,
        '221130':116,
        '221213':117,
        '230113':118,
        '230118':119,
        '230119':120,
        '230120':121,
        #'230124':122,
        #'230125':123,
        #'230126':124,
        #'230127':125,
        #'230130':126,
        #'230131':127
}

events_id = {'start_000/w/still/small/man':240+1,'start_100/w/move/small/man':32+1, 'start_010/w/still/big/man':64+1,
            'start_110/w/move/big/man':96+1, 'start_001/w/still/small/nat':128+1, 'start_101/w/move/small/nat':160+1, 
            'start_011/w/still/big/nat':192+1, 'start_111/w/move/big/nat':224+1,'start_000/p/still/small/man':240+2,
            'start_100/p/move/small/man':32+2, 'start_010/p/still/big/man':64+2, 'start_110/p/move/big/man':96+2,
            'start_001/p/still/small/nat':128+2, 'start_101/p/move/small/nat':160+2, 'start_011/p/still/big/nat':192+2, 
            'start_111/p/move/big/nat':224+2}

participant_arr=list(Part_info.keys())

Category = {
        "11":"move",
        "12":"still",
        "21":"big",
        "22":"small",
        "31":"nat",
        "32":"man"
}

method={
    1:"_all_trials",
    2:"_sptrl",
    3:"_all_tr_right",
    4:"_sptrl_right"
}
print(aa)

sc = np.zeros((np.size(list(Part_info)),271,3))

for ii in range(np.size(participant_arr)):
    participant=participant_arr[ii]
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path=data_path+'/proccessed/'

    if aa==1:
        path_file = os.path.join(result_path,data_name+'_epo.fif') 
        epochs= mne.read_epochs(path_file, preload=True,verbose=True)
    elif aa==2:
        path_file = os.path.join(result_path,data_name+'_epo-right.fif') 
        epochs= mne.read_epochs(path_file, preload=True,verbose=True)
    elif aa==3:
        path_file = os.path.join(result_path,data_name+'_supertrials.fif') 
        epochs= mne.read_epochs(path_file, preload=True,verbose=True)
    else:
        path_file = os.path.join(result_path,data_name+'_supertrials-right.fif') 
        epochs= mne.read_epochs(path_file, preload=True,verbose=True)
        
    epochs.event_id=events_id
    mc = 1
    cat_arr=[11,21,31]
    for count,cat in enumerate(cat_arr):
            mod_train = 'w'
            cat_1 = mod_train+'/'+Category[str(cat)]
            cat_2 = mod_train+'/'+Category[str(cat+1)]
            train_1 = epochs[cat_1].copy()
            train_2 = epochs[cat_2].copy()
            mod_test = 'p' if mod_train == 'w' else 'w'
            cat_1 = mod_test+'/'+Category[str(cat)]
            cat_2 = mod_test+'/'+Category[str(cat+1)]
            test_1 = epochs[cat_1].copy()
            test_2 = epochs[cat_2].copy()

            epochs_train = mne.concatenate_epochs([train_1 ,train_2])
            epochs_train = epochs_train.copy().filter(0.1,30)
            epochs_train.resample(300)
            epochs_train.crop(tmin=-0.1, tmax=0.8)
            epochs_train = epochs_train.copy().apply_baseline(baseline=(-0.1, 0))
            X_train = epochs_train.get_data(picks='meg') 
            tr_1=np.unique(train_1.events[:,2])
            tr_2=np.unique(train_2.events[:,2])
            merged_events = mne.merge_events(epochs_train.events, tr_1, 1)
            merged_events = mne.merge_events(merged_events, tr_2, 2)
            epochs_train.events=merged_events
            y_train = merged_events[:,2]

            epochs_test = mne.concatenate_epochs([test_1 ,test_2])
            epochs_test = epochs_test.copy().filter(0.1,30)
            epochs_test.resample(300)
            epochs_test.crop(tmin=-0.1, tmax=0.8)
            epochs_test = epochs_test.copy().apply_baseline(baseline=(-0.1, 0))
            X_test = epochs_test.get_data(picks='meg')
            tst_1=np.unique(test_1.events[:,2])
            tst_2=np.unique(test_2.events[:,2])
            merged_events = mne.merge_events(epochs_test.events, tst_1, 1)
            merged_events = mne.merge_events(merged_events, tst_2, 2)
            epochs_test.events=merged_events
            y_test = merged_events[:,2] 

            score_methods = make_scorer(accuracy_score)
            clf = make_pipeline(Vectorizer(),StandardScaler(),  
                            LinearModel(sklearn.svm.SVC(kernel = 'linear')))
        
            #time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
            time_decod = SlidingEstimator(clf, n_jobs=-1, scoring=score_methods, verbose=True)
            time_decod.fit(X_train, y_train)
            sc[ii,:,count] = time_decod.score(X_test, y_test)

np.save(result_all_path+'sc_CrossClass_w_p', sc)
                    

