import mne
from mne.channels import make_standard_montage
from sklearn.metrics.pairwise import euclidean_distances
import os.path as op
from init_y import *

participant_arr=list(Part_info.keys())
participant=participant_arr[0]

data_path = '/rds/projects/k/kowalcau-opm-recordings/MEG_data/20230224_anna/'
data_name = 'Part_1_A_eyes_closed'
#data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'+participant+'/'
#data_name = 'full'
path_data = os.path.join(data_path,data_name +'.fif') 
data_raw = mne.io.read_raw_fif(path_data, allow_maxshield=True,preload=True,verbose=True)

raw = data_raw.copy().pick_types(meg='mag', eeg=False, eog=False)
n_channels = 30  # number of channels to select
step = int(raw.info['nchan'] / n_channels)  # calculate the step size based on the total number of channels
channel_names = [raw.ch_names[ii] for ii in range(0, raw.info['nchan'], step)]

# create a layout from the data_raw object
layout = mne.channels.find_layout(raw.info)
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
        new_channel = 90#np.random.randint(0, distances.shape[0])
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
new_array = ['' for _ in range(len(raw.ch_names))]

# Iterate through the subset and copy corresponding elements from the original array to the new array
for elem in channel_names:
    if elem in raw.ch_names:
        new_array[raw.ch_names.index(elem)] = elem
mne.viz.plot_topomap(raw.get_data()[:,0],raw.info,names=new_array)



# select channels using the channel names and the MEG gradiometer channel type
#selection = mne.pick_channels(raw.ch_names, include=channel_names).tolist()

# create a montage to plot the channels on the scalp surface
#montage = make_standard_montage('standard_1020')
#data_raw.pick_channels(picks=selection.tolist())
# plot the topomap of the first time point in the data with the selected channels highlighted

#mne.viz.plot_topomap(data_raw.get_data(picks=channel_names)[:,0], mne.pick_info(raw.info,sel=selection) ,names=channel_names)
#mne.viz.plot_topomap(raw.get_data()[:,0],raw.info,names=new_array)

# add a title to the figure and show it
#fig.suptitle('Selected Channels')
#mne.viz.tight_layout()
#mne.viz.show()