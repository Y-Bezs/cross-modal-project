def Cat_all(sub):
        aaa = 'no_ica' # no ica
        xx = 1
        suffics = { 1:'no_max',
                2:'max_wo_head',
                3:'max_w_head'}
        for xx in [1,2,3]:
                participant_arr=list(Part_info.keys())
                participant=participant_arr[sub-1]
                data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant
                result_path=data_path+'/proccessed/'+suffics[xx]
                old=1 if Part_info[participant]<109 else 0
                data_name = 'full'

                if aaa == 'no_ica':
                        ending = '_ann-1-wo-bad-chn.fif' if xx==1 else '_ann-1.fif'
                else : 
                        ending = 'ica-1.fif'

                path_data = os.path.join(result_path,data_name +'_'+suffics[xx] + ending) 
                data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

                #events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)))                
                if old==1:
                        result_path_eve=data_path+'/proccessed/'
                        filename_events = op.join(result_path_eve,data_name + '_eve-all-new' +'.fif')
                        events = mne.read_events(filename_events, verbose=True)
                else:
                        events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step') 
                        filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')
                        mne.write_events(filename_events, events,verbose=True,overwrite=True)


                cat=np.array([240, 128, 64, 32, 192, 160, 96, 224])
                onset_tr_wrd=cat+1
                onset_tr_pic=cat+2

        #check how many wrong/right answers
                qst_left, qst_right, error_count = np.where(events[:,2]==5)[0], np.where(events[:,2]==6)[0], []
                error_count += list(qst_left[events[qst_left+2,2]==8]-6)
                error_count += list(qst_right[events[qst_right+2,2]==16]-6)
                error_count += list(np.where(events[:,2]==24)[0])
                wrong_answers = np.concatenate([qst_left[events[qst_left+2,2]==8]-6, qst_right[events[qst_right+2,2]==16]-6, np.where(events[:,2]==24)[0]])
                all_qst = len(qst_left) + len(qst_right)

                mask = np.logical_or(events[:,2] == 191, events[:,2] == 127)
                ind = np.where(mask == True)

                fig1=plt.hist(error_count, bins=ind[0], edgecolor='k')
                plt.xlabel('runs')
                plt.ylabel('errors per run (%)')
                plt.title('Overall failed questions ' + str(round(len(error_count)/all_qst*100,2)) + '%')
                filename_fig = op.join(result_path, 'Error_qst.png')
                plt.savefig(filename_fig, dpi=600)
                np.save(result_path+'Error_qst', error_count)

        # exclude trials before failed questions
                events[error_count,2]=-1
                filename_events = op.join(result_path,data_name +'_'+suffics[xx]+ '_eve-right' +'.fif')
                mne.write_events(filename_events, events,verbose=True,overwrite=True)

                raw_list = list()
                events_list = list()
                path_file = os.path.join(result_path,data_name +'_'+suffics[xx] + ending) 
                raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True)

                aa = 1
                if aa==1:
                        if np.where(events[:,2]==24)[0].size==0:
                          events_id = {
                                'fix':255, 
                                'break':1,
                                'qst_left':5,
                                'qst_right':6,
                                'ans_left':8,
                                'ans_right':16,
                                'start_000/w':240+1,
                                'start_100/w':32+1, 
                                'start_010/w':64+1, 
                                'start_110/w':96+1, 
                                'start_001/w':128+1,
                                'start_101/w':160+1, 
                                'start_011/w':192+1, 
                                'start_111/w':224+1, 
                                'start_000/p':240+2,
                                'start_100/p':32+2,
                                'start_010/p':64+2, 
                                'start_110/p':96+2, 
                                'start_001/p':128+2, 
                                'start_101/p':160+2, 
                                'start_011/p':192+2,
                                'start_111/p':224+2, 
                                'words':191,'pictures':127 }
                        else:
                          events_id = {
                                'fix':255, 
                                'break':1,
                                'qst_left':5,
                                'qst_right':6,
                                'ans_left':8,
                                'ans_right':16,
                                'start_000/w':240+1,
                                'start_100/w':32+1, 
                                'start_010/w':64+1, 
                                'start_110/w':96+1, 
                                'start_001/w':128+1,
                                'start_101/w':160+1, 
                                'start_011/w':192+1, 
                                'start_111/w':224+1, 
                                'start_000/p':240+2,
                                'start_100/p':32+2,
                                'start_010/p':64+2, 
                                'start_110/p':96+2, 
                                'start_001/p':128+2, 
                                'start_101/p':160+2, 
                                'start_011/p':192+2,
                                'start_111/p':224+2, 
                                'words':191,'pictures':127}

                raw_list.append(raw)
                events_list.append(events)
                raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
                del raw_list

                epochs = mne.Epochs(raw,
                        events, events_id,
                        tmin=-0.40 , tmax=1,
                        baseline=None,
                        proj=True,
                        picks = 'all',
                        detrend = 1,
                        #reject=reject,
                        reject_by_annotation=True,
                        preload=True,
                        verbose=True)
                
                fig=epochs.plot_drop_log()
                filename_fig = op.join(result_path, 'drop-trials.png')
                fig.savefig(filename_fig, dpi=600)
                path_outfile = os.path.join(result_path,data_name+'_'+suffics[xx]+'_epo-right.fif') 
                epochs.save(path_outfile,overwrite=True)

                pic=mne.event.match_event_names(event_names=events_id,keys=['p'])
                word=mne.event.match_event_names(event_names=events_id,keys=['w'])

                evoked_pic= epochs[pic].copy().average(method='mean').filter(0.1, 30).crop(-0.1,0.4)
                evoked_word= epochs[word].copy().average(method='mean').filter(0.1, 30).crop(-0.1,0.4)

                #evoked_word.copy().apply_baseline(baseline=(-0.1, 0))
                #fig1=evoked_word.copy().pick_types(meg='mag').plot_topo(title = 'Magnetometers/WORDS')
                #filename_fig = op.join(result_path, 'TOPO_mag_words.png')
                #fig1.savefig(filename_fig, dpi=600)

                evoked_pic.copy().apply_baseline(baseline=(-0.1, 0))
                fig2=evoked_pic.copy().pick_types(meg='mag').plot_topo(title = 'Magnetometers/PICTURES')
                filename_fig = op.join(result_path, 'TOPO_mag_pic'+aaa+'.png')
                fig2.savefig(filename_fig, dpi=600)
                
                #fig5=evoked_pic.plot()
                #filename_fig = op.join(result_path, 'ERP_amp_pictures.png')
                #fig5.savefig(filename_fig, dpi=600)

                #evoked_word.copy().apply_baseline(baseline=(-0.1, 0))
                #fig3=evoked_word.copy().pick_types(meg='grad').plot_topo(title = 'Gradiometers/WORDS')
                #filename_fig = op.join(result_path, 'TOPO_grad_words.png')
                #fig3.savefig(filename_fig, dpi=600)

                #evoked_pic.copy().apply_baseline(baseline=(-0.1, 0))
                #fig3=evoked_pic.copy().pick_types(meg='grad').plot_topo(title = 'Gradiometers/PICTURES')
                #filename_fig = op.join(result_path, 'TOPO_grad_pic.png')
                #fig3.savefig(filename_fig, dpi=600)

                ERP_all=np.mean(evoked_pic.pick_types(meg='mag').get_data(),axis=0)
                fig4=plt.figure()
                plt.plot(evoked_pic.times,ERP_all)
                filename_fig = op.join(result_path, 'ERP_all'+aaa+'.png')
                fig4.savefig(filename_fig, dpi=600)
                 
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
