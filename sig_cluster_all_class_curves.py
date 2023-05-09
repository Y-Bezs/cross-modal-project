import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats as stats
from init_y import *

# %% read data

data_path =r'Y:/cross_modal_project/Across_participants/Category_w_time/'
participant_arr=list(Part_info.keys())

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

xx=1  # choose suffics
sensor ='mag'
time_in = '_50' # time imbedded 

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[1]+'_'+ sensor + time_in


file_name_time = 'times' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[1]+'/' + file_name_time)
time = np.load(path_file)


path_to_save = data_path + 'results/' 
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

color_v={
       0:'blue',
       1:'green',
       }

mod={
       0:'WORDS',
       1:'PICTURES',
       }

Category = {
      11:"move",
      12:"still",
      21:"big",
      22:"small",
      31:"nat",
      32:"man"
}

# %% classification curves

start_ind = np.where(time==-0.100)[0][0]
end_ind   = np.where(time==0.600)[0][0]

xx=2
score_mean = np.zeros((len(participant_arr),6, end_ind-start_ind))
sensor = 'meg' # all the filter conditions

scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))
file_name2='_'+suffics[xx]+'_'+ sensor + time_in
for ii, par_name in enumerate(participant_arr):
    par=str(Part_info[par_name])
    file_name = file_name1 + par + file_name2 +'.npy'
    path_file = os.path.join(data_path+suffics[xx]+ '/' +  file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores # scores_all contains scores from all participants and has first three rows for words, lasr three for pictures.  Categories is in order of Category variable         
#score_mean_pic[iii,:,:] = np.mean(scores_all[:,3:6,start_ind:end_ind], axis = 1)
score_mean[:,:,:] = scores_all[:,:,start_ind:end_ind]

# %%
def sig_cluster(data_1):
    n_participants, n_timepoints = data_1.shape
    alpha = 0.05
    threshold = 1.7
    connectivity = None # Here you can define the connectivity if you want
    n_permutations = 1000 # Define the number of permutations for the cluster permutation test
    data = data_1-0.5

    T_obs, clusters, cluster_p_vals,_ = permutation_cluster_1samp_test(data, threshold=threshold, n_permutations=n_permutations, tail=1)
    sig_clusters = np.where(cluster_p_vals < alpha)[0]
    
    return sig_clusters,clusters
#  Print the significant clusters
times = time[start_ind:end_ind]
for aa in range(6):
    sig_clusters,clusters = sig_cluster(score_mean[:,aa,start_ind:end_ind])
    data_mean = np.mean(score_mean[:,aa,start_ind:end_ind], axis = 0)
    data_std = np.std(score_mean[:,aa,start_ind:end_ind], axis = 0)/np.sqrt(30) 
    label = 'Words' if aa < 3 else 'Picture'
    
    fig=plt.figure(aa)
    for cl in sig_clusters:
        h=plt.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red') 
    plt.plot(times,data_mean,color=color_v[aa//3], label=label)
    plt.fill_between(times, data_mean-data_std, data_mean+data_std, color=color_v[aa//3], alpha=0.2)    
    plt.axvline(.0, color='k', linestyle='-')
    plt.axvline(.6, color='k', linestyle='-')
    plt.axhline(.5, color='k', linestyle='--', label='chance')

    plt.xlabel('Times')
    plt.ylabel('AUC/scores') 
    plt.title(Category[(aa%3+1)*10+1]+' VC '+Category[(aa%3+1)*10+2]+'/ '+suffics[xx]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
    plt.ylim(0.35,0.7) 
    plt.legend(loc='lower right')
    #fig.savefig(path_to_save+Category[(aa%3+1)*10+1]+'_VC_'+Category[(aa%3+1)*10+2]+'_'+suffics[xx]+'_'+sensor+'_'+aaa, dpi=600)
    
# %%

y_data_p = np.mean(score_mean[:,3:6,start_ind:end_ind],axis = 1)
y_data_w = np.mean(score_mean[:,0:3,start_ind:end_ind],axis = 1)

fig = plt.figure()
sig_clusters,clusters = sig_cluster(y_data_p)
for cl in sig_clusters:
    h=plt.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red') 
plt.plot(times,np.mean(y_data_p,axis =0),color=color_v[0], label='Pictures')
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC/scores') 
plt.title('all categories'+'/'+suffics[xx]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
plt.ylim(0.45,0.65) 
plt.legend(loc='upper left')
fig.savefig(path_to_save+'all categories'+'_'+suffics[xx]+'_'+sensor+'_pic', dpi=600)

fig = plt.figure()
sig_clusters,clusters = sig_cluster(y_data_w)
for cl in sig_clusters:
    h=plt.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red') 
plt.plot(times,np.mean(y_data_w,axis = 0),color=color_v[1], label='Words')
plt.axvline(.0, color='k', linestyle='-')
plt.axvline(.6, color='k', linestyle='-')
plt.axhline(.5, color='k', linestyle='--', label='chance')
plt.xlabel('Times')
plt.ylabel('AUC/scores') 
plt.title('all categories'+'/'+suffics[xx]+' / '+str(len(participant_arr))+' averaged / '+sensor) 
plt.ylim(0.45,0.65) 
plt.legend(loc='upper left')
fig.savefig(path_to_save+'all categories'+'_'+suffics[xx]+'_'+sensor+'_words', dpi=600)

#fig.savefig(path_to_save+'all categories'+'_'+suffics[xx]+'_'+sensor+'_'+aaa, dpi=600)








