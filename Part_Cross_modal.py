def Cat_all(sub):

    suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}

    sensor = ('meg','grad','mag')
    delta_T = 50
    participant_arr=list(Part_info.keys())
    participant=participant_arr[sub-1]

    Category = {
        "11":"move",
        "12":"still",
        "21":"big",
        "22":"small",
        "31":"nat",
        "32":"man"}



    def run_category(mod, cat, Category, participant, delta_T, sens,xx,method):

            cat_1 = mod+'/'+Category[str(cat)]
            cat_2 = mod+'/'+Category[str(cat+1)]
            train_1 = epochs[cat_1].copy()
            train_2 = epochs[cat_2].copy()
            mod_test = 'p' if mod == 'w' else 'w' 
            cat_1 = mod_test+'/'+Category[str(cat)]
            cat_2 = mod_test+'/'+Category[str(cat+1)]
            test_1 = epochs[cat_1].copy()
            test_2 = epochs[cat_2].copy()
            epochs_train = mne.concatenate_epochs([train_1 ,train_2])
            epochs_train = epochs_train.copy().apply_baseline(baseline=(-0.1, 0))
            X_train = epochs_train.get_data(picks=sens) 

            # Get the dimensions of the original matrix
            num_conditions, num_channels, num_timepoints = X_train.shape
            X_train_lagged = np.zeros((num_conditions, num_channels*num_lags, num_timepoints-num_lags))
            for i in range(num_timepoints-num_lags):
                lagged_data = np.concatenate([X_train[:, :, i+j] for j in range(num_lags)], axis=1)
                X_train_lagged[:, :, i] = lagged_data.copy()

            tr_1=np.unique(train_1.events[:,2])
            tr_2=np.unique(train_2.events[:,2])
            merged_events = mne.merge_events(epochs_train.events, tr_1, 1)
            merged_events = mne.merge_events(merged_events, tr_2, 2)
            epochs_train.events=merged_events
            y_train = merged_events[:,2]

            epochs_test = mne.concatenate_epochs([test_1 ,test_2])
            epochs_test = epochs_test.copy().apply_baseline(baseline=(-0.1, 0))
            X_test = epochs_test.get_data(picks=sens)

            num_conditions, num_channels, num_timepoints = X_test.shape
            X_test_lagged = np.zeros((num_conditions, num_channels*num_lags, num_timepoints-num_lags))
            for i in range(num_timepoints-num_lags):
                lagged_data = np.concatenate([X_test[:, :, i+j] for j in range(num_lags)], axis=1)
                X_test_lagged[:, :, i] = lagged_data.copy()

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
            time_decod.fit(X_train_lagged, y_train)
            scores = time_decod.score(X_test_lagged, y_test)

            row_idx = (mod=='p')*3 + (cat//10-1) 
            scores_all[row_idx, :] = scores
            times_all[:] = epochs_test.times[0:(num_timepoints-num_lags)]

            fig=plt.figure(row_idx)
            plt.plot(times_all, scores, label='score')
            plt.axhline(.5, color='k', linestyle='--', label='chance')
            plt.xlabel('Times')
            plt.ylabel('AUC')  # Area Under the Curve
            plt.legend()
            plt.axvline(.0, color='k', linestyle='-')
            plt.title(cat_1+' vc '+cat_2+method+'_train_'+mod)
            #plt.show()
            
            filename_fig = op.join(path_to_save, 'train_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+method+str(delta_T)+'.png')
            fig.savefig(filename_fig, dpi=600)
            plt.close(fig)

    for xx in [2]:
        for sens in ['meg']:
            method = '_' + suffics[xx] + '_' + sens

            result_all_path='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/Across_participants/Cross_Class/'+ suffics[xx]+'/'
            data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
            result_path=data_path+'proccessed/' + suffics[xx]
            path_to_save=result_path + '/cross_class/'

            data_name='full'
            if not os.path.exists(result_all_path):
                os.makedirs(result_all_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            path_file = os.path.join(result_path, data_name + '_' + suffics[xx] + '_supertrials-right.fif') 
            epochs_raw = mne.read_epochs(path_file, preload=True, verbose=True)     
            epochs_raw.event_id=events_id
            epochs=epochs_raw.copy().filter(1,30).crop(tmin=-0.1, tmax=0.7)

            T_full = epochs.resample(500).times
            num_lags = round(delta_T/(T_full[2]-T_full[1])/1000)
            scores_all = np.zeros((len(['w', 'p']) * len([11, 21, 31]), len(T_full)-num_lags))
            times_all = np.zeros(len(T_full)-num_lags)

            results = Parallel(n_jobs=-1)(delayed(run_category)(mod, cat, Category, participant, delta_T, sens, xx, method) for mod in ['w', 'p'] for cat in [11, 21, 31])
   
            np.save(result_all_path + 'scores_move_big_nat_W_P_' + str(Part_info[participant]) + method + '_'+ str(delta_T), scores_all)
            np.save(result_all_path + 'times_' + method + '_'+ str(delta_T), times_all)
            np.save(result_path + 'scores_move_big_nat_W_P_' + str(Part_info[participant]) + method + '_'+ str(delta_T), scores_all)
            np.save(result_path + 'times_' + method + '_'+ str(delta_T), times_all)

if __name__ == "__main__":
    import sys
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
    from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
    from init_y import *
    from joblib import delayed, Parallel
    Cat_all(int(sys.argv[1]))