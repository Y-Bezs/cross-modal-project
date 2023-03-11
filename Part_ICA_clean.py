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

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'

Part_info_ICA={
        '221107':[2,4],
        '221110':[3,5],
        '221114':[2,5,13],
        '221121':[2,24],
        '221124':[1,10],
        '221125':[2,5],
        '221128':[1,23],
        '221129':[0,12],
        '221130':[16,22],
        '221213':[0,1,2],
        '230113':[1,2,7],
        '230118':[1,3,12],
        '230119':[1,23,24],
        '230120':[0,3,5],
        '230124':[3,4],
        #'230125':[0,2,11], #15 bad?
        #'230126':[124], #bad?
        '230127':[13,12],
        '230130':[0,2],
        '230131':[0,1],
        '230202':[1,7,9], #20
        '230206':[17,22],
        '230207':[3,4,9],
        '230208':[12,16],
        '230209':[1,2,10],
        #'230214':[16,27], #25 bad?
        '230215':[9,18,26],
        '230217':[2,12,17], #not done
        '230216':[2,10,23],
        '230222':[8,20],
        '230223':[9,19], #30
        '230224':[9,11,17],
        '230227':[0,2]
}
participant_arr=list(Part_info_ICA.keys())
for participant in ['230217']:#Part_info:
        data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant
        result_path=data_path +'/proccessed'#/w_head_movement'
        data_name = 'full'
        path_data = os.path.join(result_path,data_name +'_ann-1.fif') 
        data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)


        path_data = os.path.join(result_path,data_name +'_resmpl-1.fif') 
        data_raw_resmpl = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

        path_file = os.path.join(result_path,'_ica-1.fif') 
        ica=mne.preprocessing.read_ica(path_file, verbose=None)

        ica.exclude = Part_info_ICA[participant]
        path_file = os.path.join(result_path,data_name + '_ann-1' + '.fif') 
        path_outfile = os.path.join(result_path,data_name +'_ica-1' + '.fif') 
        raw_ica = mne.io.read_raw_fif(path_file,allow_maxshield=True,verbose=True,preload=True)   
        ica.apply(raw_ica)
        raw_ica.save(path_outfile,overwrite=True) 
#data_raw=raw_ica