def Cat_all(sub):
    xx = 1
    suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}
    participant_arr=list(Part_info.keys())
    participant=participant_arr[sub-1]
    if Part_info[participant]<109:
        old=1
    else:
        old=0
    def ica_components(old,participant,xx):
        data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
        result_path_common=data_path+'proccessed/'
        result_path = result_path_common + suffics[xx]

        data_name = 'full'
        path_data = os.path.join(result_path,data_name +'_'+suffics[xx] + '_ann-1.fif') 
        data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

    #%% ICA
        data_raw_resmpl = data_raw.copy().pick_types(meg=True)
        data_raw_resmpl.resample(200)
        data_raw_resmpl.filter(1, 40)
        path_outfile = os.path.join(result_path,data_name +'_resmpl-1' + '.fif') 
        data_raw_resmpl.save(path_outfile,overwrite=True)

        ica = ICA(method='fastica',
                random_state=97,
                n_components=30,
                verbose=True)
        ica.fit(data_raw_resmpl,
            verbose=True)

        path_outfile = os.path.join(result_path, '_'+suffics[xx] +'_ica-1' + '.fif') 
        ica.save(path_outfile,overwrite=True)

        fig2=ica.plot_components()
        filename_fig = op.join(result_path,'ICA_components_1.png')
        fig2[0].savefig(filename_fig, dpi=600)
        filename_fig = op.join(result_path,'ICA_components_2.png')
        fig2[1].savefig(filename_fig, dpi=600)
        #filename_fig = op.join(result_path,'ICA_components_3.png')
        #fig2[2].savefig(filename_fig, dpi=600)        

        fig1=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(0,16),start=1130, stop=1170)
        filename_fig = op.join(result_path,'ICA_sourses_1.png')
        fig1.savefig(filename_fig, dpi=600)

        fig3=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(16,30),start=1130, stop=1170)
        filename_fig = op.join(result_path,'ICA_sourses_2.png')
        fig3.savefig(filename_fig, dpi=600)

        #fig4=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(39,58),start=1130, stop=1170)
        #filename_fig = op.join(result_path,'ICA_sourses_3.png')
        #fig4.savefig(filename_fig, dpi=600)

    results = Parallel(n_jobs=-1)(delayed(ica_components)(old,participant,xx) for xx in [1])

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
#%%plot eog

#eog_epochs=mne.preprocessing.create_eog_epochs(data_raw,ch_name='EOG002',thresh=0.0002)
#eog_epochs_a=eog_epochs.average()
#eog_epochs_a.plot_joint()

#eog_evoked = mne.preprocessing.create_eog_epochs(data_raw).average()
#eog_evoked.apply_baseline(baseline=(None, -0.2))
#fig3=eog_evoked.plot_joint()
#filename_fig = op.join(result_path,'EOG_GRad_before_ICA.png')
#fig3[0].savefig(filename_fig, dpi=600)
#filename_fig = op.join(result_path,'EOG_Mag_before_ICA.png')
#fig3[1].savefig(filename_fig, dpi=600)

#ecg_evoked = mne.preprocessing.create_ecg_epochs(data_raw).average()
#ecg_evoked.apply_baseline(baseline=(None, -0.2))
#fig4=ecg_evoked.plot_joint()
#filename_fig = op.join(result_path,'ECG_before_ICA.png')
#fig4.savefig(filename_fig, dpi=600)