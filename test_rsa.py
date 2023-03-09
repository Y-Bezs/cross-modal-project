import rsatoolbox
from init_y import *
sub=1 

participant_arr=['221128']
result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/Category/'

#data_path =r'Z:/cross_modal_project/221130/'
#result_path=r'Z:/cross_modal_project/221130/proccessed/'
data_name = 'full'
participant=participant_arr[sub-1]

data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
result_path=data_path + 'proccessed/'

mod = 'w'
picks = ['mag','grad']

path_file = os.path.join(result_path,data_name+'_supertrials.fif') 
epochs_raw = mne.read_epochs(path_file, preload=True,verbose=True)
epochs_raw.filter(1,30)
epochs_raw.resample(300)
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

from typing import List, Optional
from scipy.spatial.distance import squareform

def plot_rdm_movie(rdms_data: rsatoolbox.rdm.RDMs,
                   descriptor: str,
                   n_t_display:int = 20, #
                   fig_width: Optional[int] = None,
                   timecourse_plot_rel_height: Optional[int] = None,
                   time_formatted: Optional[List[str]] = None,
                   colored_conditions: Optional[list] = None,
                   plot_individual_dissimilarities: Optional[bool] = None,
                   ):
    """ plots the RDM movie for a given descriptor

    Args:
        rdms_data (rsatoolbox.rdm.RDMs): rdm movie
        descriptor (str): name of the descriptor that created the rdm movie
        n_t_display (int, optional): number of RDM time points to display. Defaults to 20.
        fig_width (int, optional):  width of the figure (in inches). Defaults to None.
        timecourse_plot_rel_height (int, optional): height of the timecourse plot (relative to the rdm movie row).
        time_formatted (List[str], optional): time points formatted as strings.
            Defaults to None (i.e., rdms_data.time_descriptors['time'] is considered to be in seconds)
        colored_condiitons (list, optional): vector of pattern condition names to dissimilarities according to a categorical model on colored_conditions Defaults to None.
        plot_individual_dissimilarities (bool, optional): whether to plot the individual dissimilarities. Defaults to None (i.e., False if colored_conditions is notNone, True otherwise).

    Returns:
        Tuple[matplotlib.figure.Figure, npt.ArrayLike, collections.defaultdict]:

        Tuple of
            - Handle to created figure
            - Subplot axis handles from plt.subplots.
    """
    # create labels
    time = rdms_data.rdm_descriptors['time']
    unique_time = np.unique(time)
    time_formatted = time_formatted or ['%0.0f ms' % (np.round(x*1000,2)) for x in unique_time]
    whole_dis=np.array(rdms_data.pattern_descriptors['whole_dis'])
    descr=whole_dis[range(0,48,6)]
    n_dissimilarity_elements = rdms_data.dissimilarities.shape[1]

    # color mapping from colored conditions
    unsquareform = lambda a: a[np.nonzero(np.triu(a, k=1))]
    if colored_conditions is not None:
        plot_individual_dissimilarities = False if plot_individual_dissimilarities is None else plot_individual_dissimilarities
        unsquare_idx = np.triu_indices(n_dissimilarity_elements, k=1)
        pairwise_conds = unsquareform(np.array([[{c1, c2} for c1 in colored_conditions] for c2 in colored_conditions]))
        pairwise_conds_unique = np.unique(pairwise_conds)
        cnames = np.unique(colored_conditions)
        color_index = {f'{list(x)[0]} vs {list(x)[1]}' if len(list(x))==2 else f'{list(x)[0]} vs {list(x)[0]}': pairwise_conds==x for x in pairwise_conds_unique}
    else:
        color_index = {'': np.array([True]*n_dissimilarity_elements)}
        plot_individual_dissimilarities = True

    colors = plt.get_cmap('turbo')(np.linspace(0, 1, len(color_index)+1))

    # how many rdms to display
    t_display_idx = (np.round(np.linspace(0, len(unique_time)-1, min(len(unique_time), n_t_display)))).astype(int)
    t_display_idx = np.unique(t_display_idx)
    n_t_display = len(t_display_idx)

    # auto determine relative sizes of axis
    timecourse_plot_rel_height = timecourse_plot_rel_height or n_t_display // 3
    base_size = 40 / n_t_display if fig_width is None else fig_width / n_t_display

    # figure layout
    fig = plt.figure(constrained_layout=True, figsize=(base_size*n_t_display,base_size*timecourse_plot_rel_height))
    gs = fig.add_gridspec(timecourse_plot_rel_height+1, n_t_display)
    tc_ax = fig.add_subplot(gs[:-1,:])
    rdm_axes = [fig.add_subplot(gs[-1,i]) for i in range(n_t_display)]

    # plot dissimilarity timecourses
    lines = []

    dissimilarities_mean = np.zeros((rdms_data.dissimilarities.shape[1], len(unique_time)))
    dissimilarities_sem = np.zeros((rdms_data.dissimilarities.shape[1], len(unique_time)))

    for i, t in enumerate(unique_time):
        dissimilarities_mean[:, i] = np.mean(rdms_data.dissimilarities[t == time, :], axis=0)

    def _plot_mean_dissimilarities(labels=False):
        for i, (pairwise_name, idx) in enumerate(color_index.items()):
            mn = 1 - np.mean(dissimilarities_mean[idx, :],axis=0)
            se = np.std(dissimilarities_mean[idx, :],axis=0)/ np.sqrt(dissimilarities_mean.shape[0]) # se is over dissimilarities, not over subjects
            tc_ax.fill_between(unique_time, mn-se, mn+se, color=colors[i], alpha=.3)
            tc_ax.plot(unique_time, mn, color=colors[i], linewidth=2, label=pairwise_name if labels else None)

    def _plot_individual_dissimilarities():
        for i, (pairwise_name, idx) in enumerate(color_index.items()):
            tc_ax.plot(unique_time, 1 - dissimilarities_mean[idx, :].T, color=colors[i], alpha=max(1/255., 1/n_dissimilarity_elements))

    if plot_individual_dissimilarities:
        if colored_conditions is not None:
            _plot_mean_dissimilarities()
            yl = tc_ax.get_ylim()
            _plot_individual_dissimilarities()
            tc_ax.set_ylim(yl)
        else:
            _plot_individual_dissimilarities()

    if colored_conditions is not None:
        _plot_mean_dissimilarities(True)

    yl = tc_ax.get_ylim()
    for t in unique_time[t_display_idx]:
        tc_ax.plot([t,t], yl, linestyle=':', color='b', alpha=0.3)
    tc_ax.set_ylabel(f'P-corr\n({rdms_data.dissimilarity_measure})')
    tc_ax.set_xticks(unique_time)
    tc_ax.set_xticklabels([time_formatted[idx]  if idx in t_display_idx else '' for idx in range(len(unique_time))])
    dt = np.diff(unique_time[t_display_idx])[0]
    tc_ax.set_xlim(unique_time[t_display_idx[0]]-dt/2, unique_time[t_display_idx[-1]]+dt/2)
    t=("1 - " + descr[0] + "   2 - " + descr[1] + "   3 - " + descr[2] + "   4 - " + descr[3]
        + "   5 - " + descr[4] + "   6 - " + descr[5] + "   7 - " + descr[6] + "   8 - " + descr[7])
    tc_ax.text(0, 1, t, ha='left', va='top', wrap=True)

    tc_ax.legend()

    # display (selected) rdms
        # vmax = np.std(rdms_data.dissimilarities) * 2
    vmax=1
    for i, (tidx, a) in enumerate(zip(t_display_idx, rdm_axes)):
        a.imshow(np.mean(1-rdms_data.subset('time', times[tidx]).get_matrices(),axis=0), vmin=0, vmax=vmax);
        a.set_title('%0.0f ms' % (np.round(unique_time[tidx]*1000,2)))
        a.set_yticklabels(['1','2','3','4','5','6','7','8'],fontsize=6)
        a.set_yticks([3,9,15,21,27,33,39,45])
        a.set_xticklabels(['1','2','3','4','5','6','7','8'],fontsize=8)
        a.set_xticks([3,9,15,21,27,33,39,45])
        for jj in range(7):
            a.axvline((jj+1)*6, color='w', linestyle='-', linewidth =0.6 )
            a.axhline((jj+1)*6, color='w', linestyle='-', linewidth =0.6 )

    return fig, [tc_ax] + rdm_axes


whole_dis=np.array(rdms_data.pattern_descriptors['whole_dis'])
descr=whole_dis[range(0,48,6)]
nature=np.array(rdms_data.pattern_descriptors['nature'])
movement=np.array(rdms_data.pattern_descriptors['movement'])
size=np.array(rdms_data.pattern_descriptors['size'])

cat_dic = { 'nature':nature,
            'movement':movement,
            'size':size
}

for ii in range(3):
    cat_now=list(cat_dic.keys())[ii]
    fig, ax = plot_rdm_movie(
                            rdms_data,
                            descriptor=None,
                            n_t_display=10,
                            fig_width=20,
                            colored_conditions=cat_dic[cat_now]
                            )
    ax[0].set_title('data RSA movie, similarity trajectories grouped by ' + cat_now + ':    ' + mod)
    fig.savefig(result_path +'RSA_'+ cat_now[:3]+ '_'+mod+'.png')

from rsatoolbox.rdm import get_categorical_rdm
small_big_rdm = rsatoolbox.rdm.get_categorical_rdm([1 if i=='big' else 0 for i in size], 'small_vs_big')
rsatoolbox.vis.show_rdm(small_big_rdm,pattern_descriptor='index')

same=np.zeros(rdms_m.shape[0])
oposite=np.zeros(rdms_m.shape[0])
rdms_m = np.zeros(rdms_data.get_matrices().shape)
rdms_m = rdms_data.get_matrices()
anti_mask=small_big_rdm.get_matrices()
mask=1-anti_mask
for i in range(48):
    mask[0,i,i]=0
for tt in range(rdms_m.shape[0]):
    rdms_m_t = 1-rdms_m[tt,:,:]
    same[tt]=np.mean(np.multiply(rdms_m_t,mask))
    oposite[tt]=np.mean(np.multiply(rdms_m_t,anti_mask))


colored_conditions = size
time = rdms_data.rdm_descriptors['time']
unique_time = np.unique(time)
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
        print(pairwise_name)
fig=plt.figure()
plt.plot(times,same*2)
plt.plot(times,oposite)
plt.plot(times,mn)

newpath = result_path + '/rsa_matrices_w' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
def str(a):
    return "{:.3f}".format(a)

for ii in range(0,len(times),20):
    rdm_m = 1-rdms_data.subset('time', times[ii]).get_matrices()
    fig=plt.figure()   
    plt.imshow(rdm_m[0,:,:])
    plt.xticks([3,9,15,21,27,33,39,45],whole_dis[range(0,48,6)],rotation=40)
    plt.yticks([3,9,15,21,27,33,39,45],whole_dis[range(0,48,6)],rotation=0)
    plt.title('time: ' + str(times[ii]) + 'ms')
    for jj in range(7):
        plt.axvline((jj+1)*6, color='w', linestyle='-', linewidth =0.2 )
        plt.axhline((jj+1)*6, color='w', linestyle='-', linewidth =0.2)
    plt.colorbar()
    filename_fig = op.join(newpath, 'rsa_' + str(times[ii]) + 'ms.png')
    fig.savefig(filename_fig, dpi=600)

#data_plt=rsatoolbox.data.Dataset(epochs.get_data(picks=pcks)[:,:,129],
#                        #descriptors = 'modality',
#                        obs_descriptors = obs_des,
#                       channel_descriptors = chn_des,
#                        )
#data_plt.sort_by('category') 
#rdm_plt = rsatoolbox.rdm.calc_rdm(data_plt, method='correlation', descriptor=None)#

#colored_condition = nature
#if colored_conditions is not None:
#    plot_individual_dissimilarities = False if plot_individual_dissimilarities is None else plot_individual_dissimilarities
#    unsquare_idx = np.triu_indices(n_dissimilarity_elements, k=1)
#    pairwise_conds = unsquareform(np.array([[{c1, c2} for c1 in colored_conditions] for c2 in colored_conditions]))
#    pairwise_conds_unique = np.unique(pairwise_conds)
#    cnames = np.unique(colored_conditions)
#    color_index = {f'{list(x)[0]} vs {list(x)[1]}' if len(list(x))==2 else f'{list(x)[0]} vs {list(x)[0]}': pairwise_conds==x for x in pairwise_conds_unique}


