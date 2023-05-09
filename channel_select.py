def channel_select(N_chan)

    data_pos_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/analysis/'
    np.load(data_pos_path+'pos.npy')
    # get the positions of all the channels
    pos = layout.pos
    # calculate the Euclidean distances between each pair of channels
    distances = euclidean_distances(pos)
    #MEG0721 - 90
    # select N channels with the largest minimum distance to the nearest channel
    n_channels = 35
    selected_channels = []
    while len(selected_channels) < n_channels:
        if not selected_channels:  # check if selected_channels is empty
            # randomly select a channel to start with
            # new_channel = 90#np.random.randint(0, distances.shape[0])
            new_channel = np.random.randint(0, distances.shape[0])
            selected_channels.append(new_channel)
        else:
        # calculate the minimum distance to the nearest already-selected channel for each channel
            min_distances = [min([distances[ii, jj] for jj in selected_channels]) for ii in range(distances.shape[0])]
        # select the channel with the largest minimum distance
            new_channel = min_distances.index(max(min_distances))
        # add the new channel to the selected channels
            selected_channels.append(new_channel)

    # get the channel names for the selected channels
    channel_names = [raw.ch_names[ii] for ii in selected_channels]
    return channel_names
    #new_array = ['' for _ in range(len(raw.ch_names))]

    # Iterate through the subset and copy corresponding elements from the original array to the new array
    #for elem in channel_names:
    #    if elem in raw.ch_names:
    #        new_array[raw.ch_names.index(elem)] = elem
    #mne.viz.plot_topomap(raw.get_data()[:,0],raw.info,names=new_array)



