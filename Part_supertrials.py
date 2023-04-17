from init_y import *

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'
#participant_arr=['221107','221110','221114','221117','221121','221124','221125','221128','221129','221130']
#participant_arr=['221213']

data_name = 'full'
participant_arr=list(Part_info.keys())
events_id = {
        'start_000/w/still/small/man':240+1,
        'start_100/w/move/small/man':32+1, 
        'start_010/w/still/big/man':64+1, 
        'start_110/w/move/big/man':96+1, 
        'start_001/w/still/small/nat':128+1, 
        'start_101/w/move/small/nat':160+1,
        'start_011/w/still/big/nat':192+1, 
        'start_111/w/move/big/nat':224+1,
        'start_000/p/still/small/man':240+2,
        'start_100/p/move/small/man':32+2, 
        'start_010/p/still/big/man':64+2, 
        'start_110/p/move/big/man':96+2,
        'start_001/p/still/small/nat':128+2, 
        'start_101/p/move/small/nat':160+2, 
        'start_011/p/still/big/nat':192+2,
        'start_111/p/move/big/nat':224+2
        }
suffics = { 1:'no_max',
        2:'max_wo_head',
        3:'max_w_head'}

aaa = 'no_ica'



for par in range(np.size(participant_arr)):
    if par < 3:
        arr=[1,2]
    else:
        arr=[1,2,3]

    for xx in arr:
        participant=participant_arr[par]
        data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/' + participant
        result_path=data_path + '/proccessed/'+suffics[xx]
        old=1 if Part_info[participant]<109 else 0

        if aaa == 'no_ica':
                ending = '_ann-1-wo-bad-chn.fif' if xx==1 else '_ann-1.fif'
        else : 
                ending = 'ica-1.fif'

        filename_events = op.join(result_path,data_name +'_'+suffics[xx]+ '_eve-right' +'.fif')
        events = mne.read_events(filename_events, verbose=True)

        path_file = os.path.join(result_path,data_name +'_'+suffics[xx] + ending) 
        raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True)
        events_avg=np.zeros([1,3])
        # working with matrix
        for ii in [41,42,43,44,45,46]:#]range(41,47) :   
            if old==1:
                shift = 2 if participant=='221114' else 1
            else:
                shift=2
            events_obj= events[np.where(events[:,2]==ii)[0]-shift,:]
            events_list = list()
            events_list.append(events_obj)

            raw_list = list()        
            raw_list.append(raw)
            
            events_tmp=list()
            raw, events_tmp = mne.concatenate_raws(raw_list,events_list=events_list)
            del raw_list

            t_start=-0.4
            epochs = mne.Epochs(raw,
                        events_tmp, events_id,
                        tmin=t_start , tmax=1,
                        baseline=(-0.1, 0),
                        proj=True,
                        picks = 'meg',
                        detrend = 1,
                        #reject=reject,
                        reject_by_annotation=True,
                        preload=True,
                        verbose=True)#

            my_list=[]
            my_list = list(events_id.items())
            dim = epochs.get_data('meg').shape
            tmp=np.zeros([16,dim[1],1401])
            #events_avg=np.zeros([1,3])
            for jj in range(0,16):
                tmp[jj,:]=np.mean(epochs[my_list[jj][0]].get_data(picks='meg'),axis=0)
                if ii==41 & jj==0:
                    tmp_event=epochs[my_list[jj][0]].events[0,:]
                    #events_avg=np.array(tmp_event[np.newaxis,:])
                    events_avg=np.array(tmp_event)
                else:
                    events_avg=np.append(events_avg,np.array(epochs[my_list[jj][0]].events[np.newaxis,0,:]),axis=0)

            if ii==41:
                raw_avg=tmp
            else:
                raw_avg = np.append(raw_avg,tmp,axis=0)

        events_avg=np.delete(events_avg,0,0)
        events_avg=np.array(events_avg)
        Epochs_avg=mne.EpochsArray(data=raw_avg,info=epochs['start_000/p'].info,events=events_avg.astype(int),event_id=events_id,tmin=t_start)
        #
        path_outfile = os.path.join(result_path,data_name+'_'+suffics[xx]+'_supertrials-right'+ aaa +'.fif') 
        Epochs_avg.save(path_outfile,overwrite=True)


            