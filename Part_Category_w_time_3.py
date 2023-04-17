
def Cat_all(sub):

    suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}

    sensor = ('meg','grad','mag')
    delta_T=50
    participant_arr=list(Part_info.keys())
    participant=participant_arr[sub-1]

    Category = {
        "11":"move",
        "12":"still",
        "21":"big",
        "22":"small",
        "31":"nat",
        "32":"man"}

    def run_category(mod, cat, Category, participant, delta_T, sens, xx, method):

            cat_1=mod + '/' + Category[str(cat)]
            cat_2=mod + '/' + Category[str(cat+1)]

            epochs_1 = epochs[cat_1].copy()  #choose epochs to classify
            epochs_2 = epochs[cat_2].copy()
            epochs_rs=mne.concatenate_epochs([epochs_1 ,epochs_2])
            #epochs_rs=epochs_rs_raw.copy().filter(1,30)
            epochs_rs = epochs_rs.copy().apply_baseline(baseline=(-0.1, 0))
            X = epochs_rs.get_data(picks=sens) 
            X.shape

            # Get the dimensions of the original matrix
            num_conditions, num_channels, num_timepoints = X.shape
            X_lagged = np.zeros((num_conditions, num_channels*num_lags, num_timepoints-num_lags))

            # For each channel, concatenate the original data with the lagged time points
            for i in range(num_timepoints-num_lags):
                lagged_data = np.concatenate([X[:, :, i+j] for j in range(num_lags)], axis=1)
                X_lagged[:, :, i] = lagged_data

            tr_1=np.unique(epochs_1.events[:,2])  #prepare the labels
            tr_2=np.unique(epochs_2.events[:,2])
            merged_events = mne.merge_events(epochs_rs.events, tr_1, 1)
            merged_events = mne.merge_events(merged_events, tr_2, 2)
            epochs_rs.events=merged_events
            y = merged_events[:,2]  #labels

            clf = make_pipeline(Vectorizer(),StandardScaler(),  
                            LinearModel(sklearn.svm.SVC(kernel = 'linear'))) 
            time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)   
            scores = cross_val_multiscore(time_decod, X_lagged, y, cv=5, n_jobs=-1)
            scores = np.mean(scores, axis=0)

            row_idx = (mod=='p')*3 + (cat//10-1) 
            scores_all[row_idx, :] = scores
            times_all[:] = epochs_rs.times[0:(num_timepoints-num_lags)]

            fig=plt.figure(row_idx)
            plt.plot(times_all, scores, label='score')
            plt.axhline(.5, color='k', linestyle='--', label='chance')
            plt.xlabel('Times')
            plt.ylabel('AUC')  # Area Under the Curve
            plt.legend()
            plt.axvline(.0, color='k', linestyle='-')
            plt.title(cat_1+' vc '+cat_2+method)
            plt.show()
            
            filename_fig = op.join(path_to_save, mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+method+str(delta_T)+'.png')
            fig.savefig(filename_fig, dpi=600)
            

    for xx in [3]:
        for sens in ['meg','mag','grad']:
            method = '_' + suffics[xx] + '_' + sens

            result_all_path='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/Across_participants/Category_w_time/'+ suffics[xx]+'/'
            data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
            result_path=data_path+'proccessed/' + suffics[xx]
            path_to_save=result_path + '/category_w_time/'

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
        
            np.save(result_all_path + 'scores_move_big_nat_W_P_' + str(Part_info[participant]) + method+'_'+str(delta_T), scores_all)
            np.save(result_all_path + 'times' + method +'_'+ str(delta_T), times_all)
            np.save(path_to_save + 'scores_move_big_nat_W_P_' + str(Part_info[participant]) + method+'_'+str(delta_T), scores_all)
            np.save(path_to_save + 'times' + method +'_'+ str(delta_T), times_all)
#np.save(result_path+'scores_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+method[aa], scores)
#np.save(result_all_path+'scores_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+str(Part_info[participant])+method[aa], scores)
#np.save(result_all_path+'time_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+str(Part_info[participant])+method[aa], epochs_rs.times)
                 
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
    from init_y import *
    from joblib import delayed, Parallel
    Cat_all(int(sys.argv[1]))

    #if Part_info[participant]<109:
    #    old=1
    #else:
    #    old=0

   # data_name = 'full'
   # path_data = os.path.join(result_path,data_name +'_ann-1.fif') 
   # data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

    #if old==1:
    #        filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')
    #else:
    #filename_events = op.join(result_path,data_name + '_eve-right' +'.fif')
    #events = mne.read_events(filename_events, verbose=True)

