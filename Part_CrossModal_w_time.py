from init_y import *
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score

result_all_path='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/Across_participants/CrossClass_w_time'
if not os.path.exists(result_all_path):
    os.makedirs(result_all_path)

aa = 4
sub = 1

participant_arr=list(Part_info.keys())

Category = {
        "11":"move",
        "12":"still",
        "21":"big",
        "22":"small",
        "31":"nat",
        "32":"man"}

method={
    1:"_all_trials",
    2:"_sptrl",
    3:"_all_tr_right",
    4:"_fltr_1_30_"}


for participant in participant_arr[0:14]:
    data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
    result_path=data_path+'/proccessed/w_head_movement'

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

    delta_T=50
    T_full = epochs.resample(600).crop(tmin=-0.1, tmax=0.8).times
    num_lags = round(delta_T/(T_full[2]-T_full[1])/1000)
    scores_all = np.zeros((len(['w', 'p']) * len([11, 21, 31]), len(T_full)-num_lags))
    times_all = np.zeros(len(T_full)-num_lags)

    def run_category(mod, cat, Category, method, delta_T):
            num_lags = round(delta_T/(T_full[2]-T_full[1])/1000)

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
            epochs_train = epochs_train.copy().filter(1,30)
            epochs_train.resample(600)
            epochs_train.crop(tmin=-0.1, tmax=0.8)
            epochs_train = epochs_train.copy().apply_baseline(baseline=(-0.1, 0))
            X_train = epochs_train.get_data(picks='meg') 

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
            epochs_test = epochs_test.copy().filter(1,30)
            epochs_test.resample(600)
            epochs_test.crop(tmin=-0.1, tmax=0.8)
            epochs_test = epochs_test.copy().apply_baseline(baseline=(-0.1, 0))
            X_test = epochs_test.get_data(picks='meg')

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
            plt.title(cat_1+' vc '+cat_2+method[aa]+'_train_'+mod)
            #plt.show()
            
            filename_fig = op.join(result_path, 'train_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+method[aa]+str(delta_T)+'.png')
            fig.savefig(filename_fig, dpi=600)
            plt.close(fig)

    results = Parallel(n_jobs=-1)(delayed(run_category)(mod, cat, Category, method,delta_T) for mod in ['w', 'p'] for cat in [11, 21, 31])
    np.save(result_all_path + 'scores_all_move_big_nat_W_P_' + str(Part_info[participant]) + method[aa]+str(delta_T), scores_all)
    np.save(result_all_path + 'times_all_' + method[aa]+str(delta_T), times_all)
    np.save(result_path + 'scores_all_move_big_nat_W_P_' + str(Part_info[participant]) + method[aa]+str(delta_T), scores_all)
    np.save(result_path + 'times_all_' + method[aa] + str(delta_T), times_all)

