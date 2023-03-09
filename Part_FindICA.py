from init_y import *

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'
participant_arr=['230202','230206','230207','230208','230209','230214','230215','230216']
#Part_info=['221213']

#participant_arr=list(Part_info.keys())

for participant in ['230227']:#participant_arr:#Part_info:
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path= data_path+'proccessed/w_head_movement/'
    if Part_info[participant]<109:
        old=1
    else:
        old=0

    data_name = 'full'
    path_data = os.path.join(result_path,data_name +'_ann-1.fif') 
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

    path_outfile = os.path.join(result_path,'_ica-1' + '.fif') 
    ica.save(path_outfile,overwrite=True)


    fig2=ica.plot_components()
    filename_fig = op.join(result_path,'ICA_components_1.png')
    fig2[0].savefig(filename_fig, dpi=600)
    filename_fig = op.join(result_path,'ICA_components_2.png')
    fig2[1].savefig(filename_fig, dpi=600)

    fig1=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(0,15),start=235, stop=255)
    filename_fig = op.join(result_path,'ICA_sourses_1.png')
    fig1.savefig(filename_fig, dpi=600)

    fig3=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(15,30),start=235, stop=255)
    filename_fig = op.join(result_path,'ICA_sourses_2.png')
    fig3.savefig(filename_fig, dpi=600)

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