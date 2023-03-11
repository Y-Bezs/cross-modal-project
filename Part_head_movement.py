
def Cat_all(sub):
        data_name = 'full'
        participant_arr=list(Part_info.keys())        
        participant=participant_arr[sub-1]
        data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
        result_path=data_path + 'proccessed/'
        old = 1 if Part_info[participant]<109 else 1
        newpath = result_path + '/head_movement' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        path_data = os.path.join(data_path,data_name +'.fif') 
        raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

        chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
        print(f'cHPI coil frequencies extracted from raw: {chpi_freqs} Hz')
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)   
        hpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)

        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        np.save(newpath + 'chpi_locs_', chpi_locs)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
        np.save(newpath + 'head_locs_', head_pos)

        fig1 = mne.viz.plot_head_positions(head_pos, mode='traces')
        filename_fig = op.join(newpath, 'headmovement_trace.png')
        fig1.savefig(filename_fig, dpi=600)

        fig2 = mne.viz.plot_head_positions(head_pos, mode='field')
        filename_fig = op.join(newpath, 'headmovement_field.png')
        fig2.savefig(filename_fig, dpi=600)

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