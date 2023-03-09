def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print(b)
        a, b = b, a+b



def Cat_all(n):
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
    result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/Category/'
    Part_info={
        '1107':105,
        '1110':107,
        '1114':108,
        '1117':109,
        '1121':111,
        '1124':112,
        '1125':113,
        '1128':114,
        '1129':115,
        '1130':116
    }
    participant_arr=['1107','1110','1114','1117','1121','1124','1125','1128','1129','1130']
    participant=participant_arr[n]
    #for participant in ['1107','1110','1114','1117','1121','1125','1128','1129','1130']:
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/22'+participant+'/'
    result_path='/rds/projects/j/jenseno-opm/cross_modal_project/22'+participant+'/proccessed/'
    
    if Part_info[participant]<110:
        old=1
    else:
        old=0


    data_name = 'full'
    path_data = os.path.join(result_path,data_name +'_ann-1.fif') 
    data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)


    if old==1:
            filename_events = op.join(result_path,data_name + '_eve-all-new' +'.fif')
    else:
            filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')

    events = mne.read_events(filename_events, verbose=True)


    events_id = {'start_000/w/still/small/man':240+1,'start_100/w/move/small/man':32+1, 'start_010/w/still/big/man':64+1, 'start_110/w/move/big/man':96+1, 
        'start_001/w/still/small/nat':128+1, 'start_101/w/move/small/nat':160+1, 'start_011/w/still/big/nat':192+1, 'start_111/w/move/big/nat':224+1,
        'start_000/p/still/small/man':240+2,'start_100/p/move/small/man':32+2, 'start_010/p/still/big/man':64+2, 'start_110/p/move/big/man':96+2,
        'start_001/p/still/small/nat':128+2, 'start_101/p/move/small/nat':160+2, 'start_011/p/still/big/nat':192+2, 'start_111/p/move/big/nat':224+2}

    path_file = os.path.join(result_path,data_name+'_ica-1' + '.fif') 
    raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True)

    epochs = mne.Epochs(raw,
                    events, events_id,
                    tmin=-0.10 , tmax=0.8,
                    baseline=(-0.1, 0),
                    proj=True,
                    picks = 'all',
                    detrend = 1,
                    #reject=reject,
                    reject_by_annotation=True,
                    preload=True,
                    verbose=True)

    
    Category = {
            "11":"move",
            "12":"still",
            "21":"big",
            "22":"small",
            "31":"nat",
            "32":"man"
    }

    mc=1
    for mod in ['w','p']:
            for cat in [11,21,31]:
                    cat_1=mod+'/'+Category[str(cat)]
                    cat_2=mod+'/'+Category[str(cat+1)]

                    epochs_1 = epochs[cat_1].copy()
                    epochs_2 = epochs[cat_2].copy()
                    epochs_rs_raw=mne.concatenate_epochs([epochs_1 ,epochs_2])
                    epochs_rs=epochs_rs_raw.copy().filter(0,30)
                    epochs_rs.resample(300)
                    #epochs_rs.crop(tmin=-0.1, tmax=0.8)
                    X = epochs_rs.get_data(picks='grad') 
                    X.shape

                    tr_1=np.unique(epochs_1.events[:,2])
                    tr_2=np.unique(epochs_2.events[:,2])
                    merged_events = mne.merge_events(epochs_rs_raw.events, tr_1, 1)
                    merged_events = mne.merge_events(merged_events, tr_2, 2)
                    epochs_rs.events=merged_events
                    y = merged_events[:,2]

                    clf = make_pipeline(Vectorizer(),StandardScaler(),  
                                    LinearModel(sklearn.svm.SVC(kernel = 'linear'))) 
                    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)   
                    scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
                    scores = np.mean(scores, axis=0)

    #fig, ax = plt.subplots()
    #plt.ylim([0.35, 0.65])
                    fig=plt.figure(mc)
                    plt.plot(epochs_rs.times, scores, label='score')
                    plt.axhline(.5, color='k', linestyle='--', label='chance')
                    plt.xlabel('Times')
                    plt.ylabel('AUC')  # Area Under the Curve
                    plt.legend()
                    plt.axvline(.0, color='k', linestyle='-')
                    plt.title(cat_1+' vc '+cat_2+'(grad)')
                    plt.show()
                    mc=mc+1
                    filename_fig = op.join(result_path, mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+'.png')
                    fig.savefig(filename_fig, dpi=600)
                    np.save(result_path+'scores_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)], scores)
                    np.save(result_all_path+'scores_'+mod+'_'+Category[str(cat)]+'VC'+Category[str(cat+1)]+str(Part_info[participant]), scores)
                    
