from init_y import *
data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/student_prjct'
data_name = 'frq_40_47'
path_data = os.path.join(data_path,data_name +'.fif') 
data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

events = mne.find_events(data_raw, stim_channel='STI101', min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13))) 
events_new = np.zeros(events.shape)
k = 0
i = 0
ii = 1
while ii < (events.shape[0]-1):
    if events[ii+1,0]-events[ii,0] < 600:
        events_new[i,0] = (events[ii+1,0])
        events_new[i,2] = 2
        ii +=1
        if ii == 85:
            events_new[i+1,:] = events[ii+1,:]
        k += 1
    else:
        events_new[i,:] = events[ii,:]
    i +=1
    ii +=1

nonzero_rows = np.nonzero(events_new[:, 0])[0]
events_new = np.array(events_new[nonzero_rows, :])
events_new = events_new.astype(int)

events_id = {'frq_1':1, 
             'frq_2':2}

raw_list = list()
events_list = list()
events_ep = list()
raw_list.append(data_raw)
events_list.append(events_new)
raw, events_ep = mne.concatenate_raws(raw_list,events_list=events_list)
del raw_list
epochs = mne.Epochs(raw,
        events_ep, events_id,
        tmin=-0.40 , tmax=1,
        baseline=[-0.4,0],
        proj=True,
        picks = 'all',
        detrend = 1,
        #reject=reject,
        #reject_by_annotation=True,
        preload=True,
        verbose=True)

evoked_1= epochs['frq_1'].copy().average(method='mean')
evk1_spectrum = evoked_1.compute_psd(fmax=100)
evk1_spectrum.plot()

evoked_2= epochs['frq_2'].copy().average(method='mean')
evk2_spectrum = evoked_2.compute_psd(fmax=100)
evk2_spectrum.plot()