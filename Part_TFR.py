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
import time

#data_path =r'Z:/cross_modal_project/221024/'
#result_path=r'Z:/cross_modal_project/221024/proccessed/'
# data_path = r'THE PATH TO DATA ON YOUR LOCAL SYSTEM'

data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/221121/'
result_path='/rds/projects/j/jenseno-opm/cross_modal_project/221121/proccessed/'

data_name = 'full'
path_file = os.path.join(result_path,data_name+'_epo.fif') 
epochs = mne.read_epochs(path_file, preload=True,verbose=True)

filename_events = op.join(result_path,data_name + '_eve-all' +'.fif')
events = mne.read_events(filename_events, verbose=True)

if np.where(events[:,2]==24)[0].size==0:
    events_id = {'fix':255, 'break':1,'qst_left':5,'qst_right':6,'ans_left':8,'ans_right':16,
        'start_000/w':240+1,'start_100/w':32+1, 'start_010/w':64+1, 'start_110/w':96+1, 'start_001/w':128+1, 'start_101/w':160+1, 'start_011/w':192+1, 'start_111/w':224+1,
        'start_000/p':240+2,'start_100/p':32+2, 'start_010/p':64+2, 'start_110/p':96+2, 'start_001/p':128+2, 'start_101/p':160+2, 'start_011/p':192+2, 'start_111/p':224+2,
        'words':191,'pictures':127}
else:
        events_id = {'fix':255, 'break':1,'qst_left':5,'qst_right':6,'ans_left':8,'ans_right':16,
        'start_000/w':240+1,'start_100/w':32+1, 'start_010/w':64+1, 'start_110/w':96+1, 'start_001/w':128+1, 'start_101/w':160+1, 'start_011/w':192+1, 'start_111/w':224+1,
        'start_000/p':240+2,'start_100/p':32+2, 'start_010/p':64+2, 'start_110/p':96+2, 'start_001/p':128+2, 'start_101/p':160+2, 'start_011/p':192+2, 'start_111/p':224+2,
        'words':191,'pictures':127}

pic=mne.event.match_event_names(event_names=events_id,keys=['p'])
word=mne.event.match_event_names(event_names=events_id,keys=['w'])

evoked_pic= epochs[pic].copy().average(method='mean').filter(0.0, 30).crop(-0.1,0.4)
evoked_word= epochs[word].copy().average(method='mean').filter(0.0, 30).crop(-0.1,0.4)

freqs = np.arange(2, 31, 1)
n_cycles = freqs / 2 
time_bandwidth = 2.0

tfr_s_pic =  mne.time_frequency.tfr_multitaper(
    epochs[pic], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = 4,
    verbose=True)

tfr_s_word = mne.time_frequency.tfr_multitaper(
    epochs[word], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs= 4, 
    verbose=True)

tfr_s_word.plot_topomap(
                        tmin            = 0,
                        tmax            = 0.5, 
                        fmin            = 9, 
                        fmax            = 11,
                        baseline        = [-0.2,0], 
                        mode            = 'percent',
                        title           = 'Topographical map: words ',
                              #show            = False
                        )
filename_fig = op.join(result_path,'Topographical_map_alpha_words.png')
plt.savefig(filename_fig, dpi=600)


tfr_s_pic.plot(
    picks=['MEG1913'], 
    tmin=-0.5, tmax=1.0, 
    title='MEG1913/pic')
filename_fig = op.join(result_path,'TFR_1913_no_baseline_pic.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_word.plot(
    picks=['MEG1913'], 
    tmin=-0.5, tmax=1.0, 
    title='MEG1913/word')
filename_fig = op.join(result_path,'TFR_1913_no_baseline_word.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_pic.plot(
    picks=['MEG1913'], 
    baseline=[-0.500,-0.250], 
    mode="percent", 
    tmin=-0.5, tmax=1,
    title='MEG2112', 
    vmin=-0.75, vmax=0.75)  
filename_fig = op.join(result_path,'TFR_1913_pic.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_word.plot(
    picks=['MEG1913'], 
    baseline=[-0.500,-0.250], 
    mode="percent", 
    tmin=-0.5, tmax=1,
    title='MEG2112', 
    vmin=-0.75, vmax=0.75)  
filename_fig = op.join(result_path,'TFR_1913_word.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_word.plot_topo(
    tmin=-0.5, tmax=1.0, 
    baseline=[-0.5,-0.3], 
    mode="percent", 
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title='TFR of power <30 Hz')
filename_fig = op.join(result_path,'TFR_topo_word.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_pic.plot_topo(
    tmin=-0.5, tmax=1.0, 
    baseline=[-0.5,-0.3], 
    mode="percent", 
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title='TFR of power <30 Hz')

filename_fig = op.join(result_path,'TFR_topo_pic.png')
plt.savefig(filename_fig, dpi=600)

tfr_s_diff = tfr_s_word.copy()
tfr_s_diff.data = (tfr_s_pic.data - tfr_s_word.data)/(tfr_s_pic.data + tfr_s_word.data);
tfr_s_diff.plot_topo(
    tmin=-0.5, tmax=0.0, 
    fig_facecolor='w',
    font_color='k',
    title='Pic - Words');
filename_fig = op.join(result_path,'TFR_topo_diff.png')
plt.savefig(filename_fig, dpi=600)