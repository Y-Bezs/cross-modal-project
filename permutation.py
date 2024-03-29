import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats as stats


score_mean_pic = np.load( score_mean_pic)
    
# Define your data arrays
data_1 = score_mean_pic[0,:,:] # [31, 375]
data_2 = score_mean_pic[1,:,:] # [31, 375]

n_participants, n_timepoints = data_1.shape

# perform paired t-test at each time point
tvals, pvals = stats.ttest_rel(data_1, data_2)

# define threshold for clustering
alpha = 0.05
threshold = 2

t_vals, p_vals = stats.ttest_rel(data_1, data_2)

# Define clustering parameters
connectivity = None # Here you can define the connectivity if you want
n_permutations = 1000 # Define the number of permutations for the cluster permutation test

def paired_ttest(X, Y):
    return stats.ttest_1samp(X - Y, 0, axis=0)[0]

data = data_1 - data_2
# Perform cluster permutation test
F_obs, clusters, cluster_p_vals,_ = permutation_cluster_1samp_test(data, threshold=threshold, n_permutations=n_permutations, tail=0)

# Get significant clusters
sig_clusters = np.where(cluster_p_vals < alpha)[0]

# Print the significant clusters
fig, ax = plt.subplots(figsize=(12, 6))
times = np.arange(data_1.shape[1])
ax.plot(times, np.mean(data_1, axis=0), label='no max')
ax.plot(times, np.mean(data_2, axis=0), label='max')
ax.fill_between(times, np.mean(data_1, axis=0)-np.std(data_1,axis = 0)/np.sqrt(31), np.mean(data_1, axis=0)+np.std(data_1,axis = 0)/np.sqrt(31), alpha=0.2)
ax.fill_between(times, np.mean(data_2, axis=0)-np.std(data_2,axis = 0)/np.sqrt(31), np.mean(data_2, axis=0)+np.std(data_2,axis = 0)/np.sqrt(31), alpha=0.2)   
ax.axhline(y=0.5, color='black', linestyle='--')
for cl in sig_clusters:
    h=ax.axvspan(times[clusters[cl][0][0]], times[clusters[cl][0][-1]], alpha=0.5, color='red')
ax.legend()
plt.show()
