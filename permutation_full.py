import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats as stats
from init_y import *

# %% read data

cond = 'no_ica'
data_path =r'Y:/cross_modal_project/Across_participants/Category_w_time/'
participant_arr=list(Part_info.keys())

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

xx=2  # choose suffics
sensor ='mag'
time_in = '_50' # time imbedded 

file_name1 = 'scores_move_big_nat_W_P_'
file_name2='_'+suffics[1]+'_'+ sensor+'_'+ cond + time_in


file_name_time = 'times' +file_name2+'.npy'
path_file = os.path.join(data_path + suffics[1]+'/' + cond + '/' + file_name_time)
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

# %% take one pipleline and different sensors

start_ind = np.where(time==0)[0][0]
end_ind   = np.where(time==0.600)[0][0]

sensor ='mag'
xx=3
score_mean_pic = np.zeros((3,len(participant_arr), end_ind-start_ind))
for iii,sensor in enumerate(['mag','grad','meg']): # all the filter conditions
    scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))
    file_name2='_'+suffics[xx]+'_'+ sensor+'_'+ cond + time_in
    for ii, par_name in enumerate(participant_arr):
        par=str(Part_info[par_name])
        file_name = file_name1 + par + file_name2 +'.npy'
        path_file = os.path.join(data_path+suffics[xx]+ '/' + cond +'/'+ file_name)
        scores = np.load(path_file)
        scores_all[ii,:,:] = scores # scores_all contains scores from all participants and has first three rows for words, lasr three for pictures.  Categories is in order of Category variable         
    score_mean_pic[iii,:,:] = np.mean(scores_all[:,3:6,start_ind:end_ind], axis = 1)
    #score_mean_pic[xx-1,:,:] = scores_all[:,5,:]
    
# %%
    
# Define your data arrays
data_1 = score_mean_pic[1,:,:] # [31, 375]
data_2 = score_mean_pic[0,:,:] # [31, 375]

n_participants, n_timepoints = data_1.shape

# define threshold for clustering
alpha = 0.05
threshold = 1.7

t_vals, p_vals = stats.ttest_rel(data_1, data_2)

# Define clustering parameters
connectivity = None # Here you can define the connectivity if you want
n_permutations = 1000 # Define the number of permutations for the cluster permutation test


data = data_1-data_2
# Perform cluster permutation test
T_obs, clusters, cluster_p_vals,_ = permutation_cluster_1samp_test(data, threshold={'start': 0, 'step': 0.2}, n_permutations=n_permutations, tail=0)

# Get significant clusters
sig_clusters = np.where(cluster_p_vals < alpha)[0]
# %% Print the significant clusters


fig, ax = plt.subplots()
#times = np.arange(data_1.shape[1])
times = time[start_ind:end_ind]
for cl in sig_clusters:
    h=ax.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red')
ax.plot(times, np.mean(data_1, axis=0),'k-', label='meg')
ax.plot(times, np.mean(data_2, axis=0),'g-',label='mag')
ax.fill_between(times, np.mean(data_1, axis=0)-np.std(data_1,axis = 0)/np.sqrt(31), np.mean(data_1, axis=0)+np.std(data_1,axis = 0)/np.sqrt(31), color = 'k', alpha=0.2)
ax.fill_between(times, np.mean(data_2, axis=0)-np.std(data_2,axis = 0)/np.sqrt(31), np.mean(data_2, axis=0)+np.std(data_2,axis = 0)/np.sqrt(31), color = 'g', alpha=0.2)   
ax.axhline(y=0.5, color='black', linestyle='--')
ax.axvline(.0, color='k', linestyle='-')
ax.axvline(.6, color='k', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('AUC/scores')
ax.set_title('Pic/ '+'all categories '+' / '+str(len(participant_arr))+' averaged / '+suffics[xx]) 
ax.set_ylim(0.45,0.65)
ax.legend()
plt.show()

#fig.savefig(path_to_save+'Pic_'+'all_categories_'+'_'+suffics[xx]+'_sig_clsr'+'_1_3', dpi=600)
# %% take one sensor type and different pipleline

start_ind = np.where(time==0)[0][0]
end_ind   = np.where(time==0.600)[0][0]

sensor ='mag'
xx=1
score_mean_pic = np.zeros((3,len(participant_arr), end_ind-start_ind))
for iii,xx in enumerate([1,2]): # all the filter conditions
    scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))
    file_name2='_'+suffics[xx]+'_'+ sensor+'_'+ cond + time_in
    for ii, par_name in enumerate(participant_arr):
        par=str(Part_info[par_name])
        file_name = file_name1 + par + file_name2 +'.npy'
        path_file = os.path.join(data_path+suffics[xx]+ '/' + cond +'/'+ file_name)
        scores = np.load(path_file)
        scores_all[ii,:,:] = scores # scores_all contains scores from all participants and has first three rows for words, lasr three for pictures.  Categories is in order of Category variable         
    score_mean_pic[iii,:,:] = np.mean(scores_all[:,3:6,start_ind:end_ind], axis = 1)
    #score_mean_pic[xx-1,:,:] = scores_all[:,5,:]
    
# %%

    
# Define your data arrays
data_1 = score_mean_pic[0,:,:] # [31, 375]
data_2 = score_mean_pic[1,:,:] # [31, 375]

n_participants, n_timepoints = data_1.shape

# perform paired t-test at each time point
tvals, pvals = stats.ttest_rel(data_1, data_2)

# define threshold for clustering
alpha = 0.05
threshold = 1.6

t_vals, p_vals = stats.ttest_rel(data_1, data_2)

# Define clustering parameters
connectivity = None # Here you can define the connectivity if you want
n_permutations = 1000 # Define the number of permutations for the cluster permutation test

def paired_ttest(X, Y):
    return stats.ttest_1samp(X - Y, 0, axis=0)[0]

data = data_1-data_2
# Perform cluster permutation test
T_obs, clusters, cluster_p_vals,_ = permutation_cluster_1samp_test(data, threshold={'start': 0, 'step': 0.2}, n_permutations=n_permutations, tail=1)

# Get significant clusters
sig_clusters = np.where(cluster_p_vals < alpha)[0]
# %% Print the significant clusters


fig, ax = plt.subplots()
#times = np.arange(data_1.shape[1])
times = time[start_ind:end_ind]
for cl in sig_clusters:
    h=ax.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red', label = 'significant cluster')
ax.plot(times, np.mean(data_1, axis=0),'m-', label='no max')
ax.plot(times, np.mean(data_2, axis=0),'b-',label='max wo head')
ax.fill_between(times, np.mean(data_1, axis=0)-np.std(data_1,axis = 0)/np.sqrt(31), np.mean(data_1, axis=0)+np.std(data_1,axis = 0)/np.sqrt(31), color = 'm', alpha=0.2)
ax.fill_between(times, np.mean(data_2, axis=0)-np.std(data_2,axis = 0)/np.sqrt(31), np.mean(data_2, axis=0)+np.std(data_2,axis = 0)/np.sqrt(31), color = 'b', alpha=0.2)   
ax.axhline(y=0.5, color='black', linestyle='--')
ax.axvline(.0, color='k', linestyle='-')
ax.axvline(.6, color='k', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('AUC/scores')
ax.set_title('Pic/ '+'all categories '+' / '+str(len(participant_arr))+' averaged / '+sensor) 
ax.set_ylim(0.45,0.65)
ax.legend()
plt.show()

#fig.savefig(path_to_save+'Pic_'+'all_categories_'+'_'+sensor+'_sig_clsr'+'_1_2', dpi=600)
# %%
# %% classification curves

start_ind = np.where(time==-0.100)[0][0]
end_ind   = np.where(time==0.600)[0][0]

xx=2
score_mean_pic = np.zeros((len(participant_arr),6, end_ind-start_ind))
sensor = 'meg' # all the filter conditions

scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))
file_name2='_'+suffics[xx]+'_'+ sensor+'_'+ cond + time_in
for ii, par_name in enumerate(participant_arr):
    par=str(Part_info[par_name])
    file_name = file_name1 + par + file_name2 +'.npy'
    path_file = os.path.join(data_path+suffics[xx]+ '/' + cond +'/'+ file_name)
    scores = np.load(path_file)
    scores_all[ii,:,:] = scores # scores_all contains scores from all participants and has first three rows for words, lasr three for pictures.  Categories is in order of Category variable         
#score_mean_pic[iii,:,:] = np.mean(scores_all[:,3:6,start_ind:end_ind], axis = 1)
score_mean_pic[:,:] = scores_all[:,:,start_ind:end_ind]
    
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
# %% Print the significant clusters
times = time[start_ind:end_ind]
for aa in range(6):
    sig_clusters,clusters = sig_cluster(score_mean_pic[:,aa,start_ind:end_ind])
    data_mean = np.mean(score_mean_pic[:,aa,start_ind:end_ind], axis = 0)
    data_std = np.std(score_mean_pic[:,aa,start_ind:end_ind], axis = 0)/np.sqrt(30) 
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
fig, ax = plt.subplots()
#times = np.arange(data_1.shape[1])
times = time[start_ind:end_ind]
for cl in sig_clusters:
    h=ax.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.1, color='red')
ax.plot(times, np.mean(data_1, axis=0),'k-', label='meg')
ax.fill_between(times, np.mean(data_1, axis=0)-np.std(data_1,axis = 0)/np.sqrt(30), np.mean(data_1, axis=0)+np.std(data_1,axis = 0)/np.sqrt(30), color = 'k', alpha=0.2)
ax.axhline(y=0.5, color='black', linestyle='--')
ax.axvline(.0, color='k', linestyle='-')
ax.axvline(.6, color='k', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('AUC/scores')
ax.set_title('Pic/ '+'all categories '+' / '+str(len(participant_arr))+' averaged / '+suffics[xx]) 
ax.set_ylim(0.35,0.65)
ax.legend()
plt.show()

#fig.savefig(path_to_save+'Pic_'+'all_categories_'+'_'+suffics[xx]+'_sig_clsr'+'_1_3', dpi=600)