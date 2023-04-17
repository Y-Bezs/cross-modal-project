def Cat_all(sub):
        xx = 1
        suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}
        participant_arr=list(Part_info.keys())
        participant=participant_arr[sub-1]

        Part_info_ICA={
                '221107':([2,4,18],[2,4],[]),
                '221110':([3,4,6],[3,5],[]), #14,15
                '221114':([2,4,12],[2,5,13],[]), 
                '221121':([0,13,20],[2,26],[2,24]), 
                '221124':([1,2,9],[2,9],[1,10]), 
                '221125':([5,14,15,16,19,20,22,25],[2,3,6,8],[2,3,6,8]), #may be not for xx=1
                '221128':([2,8,20],[1,25],[1,23]),
                '221129':([0,17,20],[0,7],[0,12]),
                '221130':([8,13],[9,17],[16,22]),
                '221213':([0,1,2,7,8],[0,1,2,7],[0,1,2]), #7 xx=1
                '230113':([3,5,25],[2,4,7,8],[1,2,7,8]),
                '230118':([2,4,20],[1,5,15],[1,3,12]),
                '230119':([0,14,18],[1,16,27],[1,23,24]), 
                '230120':([1,2,6,8],[0,3,21],[0,3,4]),
                '230124':([1,4,6,8],[3,1,4],[3,4]), 
                #'230125':[0,2,11], #15 bad?
                #'230126':[124], #bad?
                '230127':([2,15,25],[8,16],[12,13]),
                '230130':([1,6,7,3],[0,1],[0,1,2]),
                '230131':([0,2,4,11,26],[0,1],[0,1]),
                '230202':([0,2,3,25,26,27,28],[2,3,6,28],[1,2,7,9]), 
                '230206':([8,27],[13,20],[17,22]),
                '230207':([3,5,11],[3,4,12],[3,4,9]),
                '230208':([3,7,15,20,24],[14,17,22],[12,16]),#18
                '230209':([0,5],[1,2],[1,2]),
                #'230214':[16,27], #25 bad? should use
                '230215':([4,5,6,7],[5,12,16,19],[7,9,18,26]), #5 here
                #'230217':([],[],[]), #?
                '230216':([2,10,23,29],[1,16,28],[1,12,28]),
                '230222':([4,6,22],[10,20],[8,20]),
                '230223':([5,20,23],[13,22],[9,19]), #30
                '230224':([1,18,21,23],[5,24,27],[9,11,17]),
                '230227':([0,1,4],[0,2],[0,2]),
                '230302':([5,7,11],[5,9,28],[4,7,24]),
                '230303':([0,2,14,16],[5,11,25],[4,7,19]), #11
                '230306':([0,6,7,16],[0,9,11,12],[0,6,7,12]),
                #'230308':([4,12,15],[1,3,6,20,21,29],[1,5,8,22,23,29]), #check my be not
                '230309':([2,3,6,10,17],[1,2,3,4],[1,2,3,6])
        }
        def clean_data(participant,xx,Part_info_ICA):
                data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant
                result_path=data_path +'/proccessed'+'/'+suffics[xx]
                data_name = 'full'
                path_data = os.path.join(result_path,data_name + '_'+suffics[xx] +'_ann-1.fif') 
                data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

                path_data = os.path.join(result_path,data_name +'_resmpl-1.fif') 
                data_raw_resmpl = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

                path_file = os.path.join(result_path,'_'+suffics[xx]+'_ica-1.fif') 
                ica=mne.preprocessing.read_ica(path_file, verbose=None)

                ica.exclude = Part_info_ICA[participant][xx-1]
                path_outfile = os.path.join(result_path,data_name +'_'+suffics[xx] + 'ica-1' + '.fif')  
                ica.apply(data_raw)
                data_raw.save(path_outfile,overwrite=True) 

        results = Parallel(n_jobs=-1)(delayed(clean_data)(participant,xx,Part_info_ICA) for xx in [2,3])
#data_raw=raw_ica

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