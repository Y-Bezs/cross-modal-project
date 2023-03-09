from init_y import *

participant='230113'

events_id = {'start_000/w/still/small/man':240+1,'start_100/w/move/small/man':32+1, 'start_010/w/still/big/man':64+1, 
        'start_110/w/move/big/man':96+1, 'start_001/w/still/small/nat':128+1, 'start_101/w/move/small/nat':160+1,
        'start_011/w/still/big/nat':192+1, 'start_111/w/move/big/nat':224+1,'start_000/p/still/small/man':240+2,
        'start_100/p/move/small/man':32+2, 'start_010/p/still/big/man':64+2, 'start_110/p/move/big/man':96+2,
        'start_001/p/still/small/nat':128+2, 'start_101/p/move/small/nat':160+2, 'start_011/p/still/big/nat':192+2,
        'start_111/p/move/big/nat':224+2}

data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
result_path=data_path+'/proccessed/'
a = False
if a:
    path_file = os.path.join(result_path,data_name+'_epo.fif') 
    epochs_all= mne.read_epochs(path_file, preload=True,verbose=True)
    epochs_all.event_id=events_id

    path_file = os.path.join(result_path,data_name+'_supertrials.fif') 
    epochs_sptrl = mne.read_epochs(path_file, preload=True,verbose=True)
    epochs_sptrl.event_id=events_id


path_file = os.path.join(result_path,data_name+'_epo-right.fif') 
epochs_right= mne.read_epochs(path_file, preload=True,verbose=True)
epochs_right.event_id=events_id

path_file = os.path.join(result_path,data_name+'_supertrials-right.fif') 
epochs_sptrl_right = mne.read_epochs(path_file, preload=True,verbose=True)
epochs_sptrl_right.event_id=events_id

if a:
    epochs_all['p'].plot_image(picks='mag', combine='mean')
    epochs_sptrl['p'].plot_image(picks='mag', combine='mean')

epochs_right['p'].plot_image(picks='mag', combine='mean')
epochs_sptrl_right['p'].plot_image(picks='mag', combine='mean')