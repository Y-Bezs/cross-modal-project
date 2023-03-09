import rsatoolbox
from init_y import *
from typing import List, Optional
from scipy.spatial.distance import squareform
from rsatoolbox.rdm import get_categorical_rdm

data_name = 'full'
participant_arr=list(Part_info.keys())
for ii in range(np.size(participant_arr)):
    participant=participant_arr[ii]

    result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/rsa/'
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path=data_path + 'proccessed/'
    picks = ['mag','grad']
    path_file = os.path.join(result_path,data_name+'_supertrials-right.fif') 
    epochs_raw = mne.read_epochs(path_file, preload=True,verbose=True)
    epochs_raw.filter(1,30)
    epochs_raw.resample(300)
    for mod in ['p','w']:
        epochs = epochs_raw[mod].copy().crop(-0.1,0.7)
        epochs.apply_function(lambda x: (x - np.mean(x) / np.std(x)))
        epochs.apply_baseline(baseline=(-0.1, 0))


        ch_names = [ch_name for ch_name, ch_type in zip(epochs.ch_names, epochs.get_channel_types()) if ch_type in picks]
        event_ids = epochs.event_id
        times = epochs.times

        rev_event_id = {v: k for k, v in epochs.event_id.items()}
        event_names = np.array([rev_event_id[i] for i in epochs.events[:, 2]])

        category = np.array([(event_name.split('/')[0][6:]) for event_name in event_names])
        modality = np.array([(event_name.split('/')[1]) for event_name in event_names])
        movement = np.array([event_name.split('/')[2] for event_name in event_names])
        size = np.array([event_name.split('/')[3] for event_name in event_names])
        nature = np.array([event_name.split('/')[4] for event_name in event_names])
        whole_dis = np.array([event_name[12:] for event_name in event_names])

        des = {'session': 0}                        # some (made up) metadata, we could also append data session-wise and crossvalidate across sessions ...
                                            # ... but we'll just pretend all data came from one session and crossvalidate across image category repetitions
        obs_des = dict( 
                    modality=modality,                           # observation descriptors
                    category=category,                       
                    movement=movement,                
                    size=size,                  
                    nature=nature,
                    whole_dis=whole_dis                         
                    )
        chn_des = {'channels': ch_names}            # channel descriptors
        tim_des = {'time': times} 

        #data = epochs.get_data(picks='meg')
        pcks='meg' if len(picks)==2 else picks[0]
        data=rsatoolbox.data.TemporalDataset(epochs.get_data(picks=pcks),
                                #descriptors = 'modality',
                                obs_descriptors = obs_des,
                                channel_descriptors = chn_des,
                                time_descriptors = tim_des)
        data.sort_by('category')  

        rdms_data = rsatoolbox.rdm.calc_rdm_movie(
                                data, # list of length n_subjects
                                method = 'correlation',
                                descriptor = None
                                )

        whole_dis=np.array(rdms_data.pattern_descriptors['whole_dis'])
        descr=whole_dis[range(0,48,6)]
        nature=np.array(rdms_data.pattern_descriptors['nature'])
        movement=np.array(rdms_data.pattern_descriptors['movement'])
        size=np.array(rdms_data.pattern_descriptors['size'])

        cat_dic = { 'nature':nature,
                    'movement':movement,
                    'size':size
        }
        time = rdms_data.rdm_descriptors['time']
        unique_time = np.unique(time)
        for ii in range(3):
            cat_now=list(cat_dic.keys())[ii]
            colored_conditions = cat_dic[cat_now]
            unsquareform = lambda a: a[np.nonzero(np.triu(a, k=1))]
            pairwise_conds = unsquareform(np.array([[{c1, c2} for c1 in colored_conditions] for c2 in colored_conditions]))
            pairwise_conds_unique = np.unique(pairwise_conds)
            color_index = {f'{list(x)[0]} vs {list(x)[1]}' if len(list(x))==2 else f'{list(x)[0]} vs {list(x)[0]}': pairwise_conds==x for x in pairwise_conds_unique}
            dissimilarities_mean = np.zeros((rdms_data.dissimilarities.shape[1], len(unique_time)))
            mn = np.zeros([3,time.shape[0]])
            for i, t in enumerate(unique_time):
                dissimilarities_mean[:, i] = np.mean(rdms_data.dissimilarities[t == time, :], axis=0)
            for i, (pairwise_name, idx) in enumerate(color_index.items()):
                mn[i,:] = 1 - np.mean(dissimilarities_mean[idx, :],axis=0)
            np.save(result_all_path+'rsa_'+mod+'_'+cat_now+str(Part_info[participant]), mn)
  


