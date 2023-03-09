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

    filename_fig = op.join(result_path,'ICA_components_2.png')


    fig1=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(0,15),start=235, stop=255)
    filename_fig = op.join(result_path,'ICA_sourses_1.png')
    
    fig3=ica.plot_sources(data_raw_resmpl,title='ICA',picks=range(15,30),start=235, stop=255)
    filename_fig = op.join(result_path,'ICA_sourses_2.png')
    