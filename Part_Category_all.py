
def Cat_all(sub):

    aa = 4
    result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/Category/'
    participant_arr=list(Part_info.keys())
    participant=participant_arr[sub-1]
    
    #participant_arr=['1107','1110','1114','1121','1124','1125','1128','1129','1130']
    #for participant in ['1107','1110','1114','1117','1121','1125','1128','1129','1130']:
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path=data_path+'proccessed/w_head_movement'
    
    path_to_save = result_path + '/category_class/w_head_movement' 
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    def read_epochs(aa, result_path, data_name):
        file_extensions = {
            1: '_epo.fif',
            2: '_epo-right.fif',
            3: '_supertrials.fif',
            4: '_supertrials-right.fif'
        }
        extension = file_extensions.get(aa, '_supertrials-right.fif')
        path_file = os.path.join(result_path, data_name + '_supertrials-right.fif') 
        epochs = mne.read_epochs(path_file, preload=True, verbose=True)
        return epochs  

    epochs = read_epochs(aa, result_path, data_name)       
    epochs.event_id=events_id

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
            4:"_head_1_30"
    }

    T_full = epochs.resample(300).crop(tmin=-0.1, tmax=0.8).times
    scores_all = np.zeros((len(['w', 'p']) * len([11, 21, 31]), len(T_full)))
    times_all = np.zeros(len(T_full))
    mc = 0
    def run_category(mod, cat, Category, method, mc):
            cat_1=mod+'/'+Category[str(cat)]
            cat_2=mod+'/'+Category[str(cat+1)]

            epochs_1 = epochs[cat_1].copy()  #choose epochs to classify
            epochs_2 = epochs[cat_2].copy()
            epochs_rs_raw=mne.concatenate_epochs([epochs_1 ,epochs_2])
            epochs_rs=epochs_rs_raw.copy().filter(1,30)
            epochs_rs.resample(300)
            epochs_rs.crop(tmin=-0.1, tmax=0.8)
            epochs_rs = epochs_rs.copy().apply_baseline(baseline=(-0.1, 0))
            X = epochs_rs.get_data(picks='meg') 
            X.shape

            tr_1=np.unique(epochs_1.events[:,2])  #prepare the labels
            tr_2=np.unique(epochs_2.events[:,2])
            merged_events = mne.merge_events(epochs_rs_raw.events, tr_1, 1)
            merged_events = mne.merge_events(merged_events, tr_2, 2)
            epochs_rs.events=merged_events
            y = merged_events[:,2]  #labels

            clf = make_pipeline(Vectorizer(),StandardScaler(),  
                            LinearModel(sklearn.svm.SVC(kernel = 'linear'))) 
            time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)   
            scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
            scores = np.mean(scores, axis=0)

            row_idx = (mod=='p')*3 + (cat//10-1) 
            scores_all[row_idx, :] = scores
            times_all[:] = epochs_rs.times

            fig=plt.figure(row_idx)
            plt.plot(epochs_rs.times, scores, label='score')
            plt.axhline(.5, color='k', linestyle='--', label='chance')
            plt.xlabel('Times')
            plt.ylabel('AUC')  # Area Under the Curve
            plt.legend()
            plt.axvline(.0, color='k', linestyle='-')
            plt.title(cat_1+' vc '+cat_2+method[aa])
            plt.show()
            
            filename_fig = op.join(path_to_save, mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+method[aa]+'.png')
            fig.savefig(filename_fig, dpi=600)
            mc += 1

    
    results = Parallel(n_jobs=-1)(delayed(run_category)(mod, cat, Category, method, mc) for mod in ['w', 'p'] for cat in [11, 21, 31])
    np.save(result_all_path + 'scores_all_move_big_nat_W_P' + str(Part_info[participant])+method[aa], scores_all)
    np.save(result_all_path + 'times_all' + method[aa], times_all)
    np.save(path_to_save + 'scores_all_move_big_nat_W_P' + str(Part_info[participant])+method[aa], scores_all)
    np.save(path_to_save + 'times_all' + method[aa], times_all)
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

