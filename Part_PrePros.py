
def Cat_all(sub):
    xx = 1
    suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}

    participant_arr=list(Part_info.keys())
    participant=participant_arr[sub-1]
    def pre_pros_method(participant,xx):
        data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
        result_path_common=data_path+'proccessed/'
        result_path = result_path_common + suffics[xx]

        if not os.path.exists(result_path):
                os.makedirs(result_path)
        
        if Part_info[participant]<109:
            old=1
        else:
            old=0

        data_name = 'full'
        path_data = os.path.join(data_path,data_name +'.fif') 
        data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)
        #data_raw=data_raw.filter(2.5,100)
        #%%  MARK BREAKS
        if old==1:
            filename_events = op.join(result_path_common,data_name + '_eve-all-new' +'.fif')
            events = mne.read_events(filename_events, verbose=True)
        else: 
            events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', \
                                    mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step') 



        annotations_break=mne.preprocessing.annotate_break(data_raw, events=events,
                                                    min_break_duration=7, 
                                                    t_start_after_previous=0.1, 
                                                    t_stop_before_next=-5, 
                                                    ignore=('bad', 'edge'))
        data_raw.set_annotations(data_raw.annotations+annotations_break)  

        #%%
        #file_sss_path =r'Z:/cross_modal_project/'
        file_sss_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'
        crosstalk_file = os.path.join(file_sss_path,'ct_sparse_SA.fif')
        cal_file = os.path.join(file_sss_path,'sss_cal_SA.dat')

        #fig1=dataW1.plot_psd(fmax=60);
        #dataW1.plot(duration=5,n_channels=58,scalings='auto')
        if participant=='230125':
            data_raw.info['bads'] = ['MEG1211','MEG0921']
        elif participant=='230125':
            data_raw.info['bads'] = ['MEG0242']
        else:
            data_raw.info['bads'] = []
            
        data_raw_check = data_raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
            data_raw_check, 
            cross_talk=crosstalk_file, 
            skip_by_annotation=('edge', 'bad_acq_skip','BAD'),
            calibration=cal_file,
            return_scores=True, 
            verbose=True)

        print('noisy =', auto_noisy_chs)
        print('flat = ', auto_flat_chs)

        data_raw.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
        #%%
        data_raw.fix_mag_coil_types() 

        fig1=data_raw.plot_psd(
                            fmax=60, 
                            n_fft = 5000,
                            reject_by_annotation = True)    
        filename_fig = op.join(result_path,'PSD_before_'+suffics[xx]+'.png')
        fig1.savefig(filename_fig, dpi=600)

        if xx == 3:
            path_file = os.path.join(result_path_common+'head_movement/', 'head_movementhead_locs_.npy')
            head_locs = np.load(path_file)  
        else: 
            head_locs = None 

        if xx == 1:
            data_fltrd=data_raw.copy().filter(0,100)
        else:
            data_raw_sss = mne.preprocessing.maxwell_filter(
                data_raw,
                cross_talk=crosstalk_file,
                calibration=cal_file,
                head_pos = head_locs, #no head position for 230217/135
                skip_by_annotation = ('edge', 'bad_acq_skip','BAD'),
                verbose=True)
            fig2=data_raw_sss.plot_psd(
                                        fmax=60, 
                                        n_fft = 5000,
                                        reject_by_annotation = True) 

            filename_fig = op.join(result_path,'PSD_filtered_'+suffics[xx]+'.png')
            fig2.savefig(filename_fig, dpi=600)
            data_fltrd=data_raw_sss.copy().filter(0,100)        

        threshold_muscle = 10  
        annotations_muscle, scores_muscle = annotate_muscle_zscore(
            data_fltrd, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
            filter_freq=[110, 140])
        data_fltrd.set_annotations(annotations_muscle+data_raw.annotations)
        path_file_results = os.path.join(result_path,data_name +'_'+suffics[xx]+'_ann-1.fif') 
        data_fltrd.save(path_file_results,overwrite=True) 
    
    results = Parallel(n_jobs=-1)(delayed(pre_pros_method)(participant,xx) for xx in [1,2,3])
    

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

#num1 = [len(np.where(events[:,2]==x)[0]) for x in cat+2]
#duration=np.empty((0))
#onset=np.empty((0))
#description=[]
#breaks_tmp=np.where(events[:,2]==1)[0]
#for i in range(0,len(breaks_tmp)-1):
#    duration=np.append(duration,events[breaks_tmp[i]+2,0]-events[breaks_tmp[i],0])
#    onset=np.append(onset,events[breaks_tmp[i],0])
#i=i+1
#duration=np.append(duration,events[breaks_tmp[i]+1,0]-events[breaks_tmp[i],0])
#onset=np.append(onset,events[breaks_tmp[i],0])
#num_breaks=len(breaks_tmp)
#orig_time = data_raw.info['meas_date']
#annototations_break = mne.Annotations(onset=onset/1000,  # in seconds
#                           duration=duration/1000,  # in seconds, too
#                           description=repeat('BAD', num_breaks),orig_time=orig_time)

#data_raw.set_annotations(annototations_break) 