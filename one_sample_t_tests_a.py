import numpy as np
import glmtools as glm
from glmtools.permutations import MaxStatPermutation, ClusterPermutation

#words = np.load('score_size_train_p.npy')
words = np.load('score_all_train_p.npy')

words = words - 0.5  # Adjust so we test against zero


sliced_array_shifted = words[:, 38:, 38:]
sliced_array = sliced_array_shifted[:,::5,::5]


data = glm.data.TrialGLMData(data=sliced_array, dim_labels=['Participants', 'Time', 'Time'])

DC = glm.design.DesignConfig()
DC.add_regressor(name='Mean', rtype='Constant')
DC.add_simple_contrasts()

design = DC.design_from_datainfo(data.info)

model = glm.fit.OLSModel(design, data)



#%% Cluster stats should be better
nperms = 1000
P = ClusterPermutation(design, data, 0, nperms, pooled_dims=(1, 2), tail=1,
                       metric='tstats', nprocesses=6, cluster_forming_threshold=2)
clusters = P.get_sig_clusters(data, [90])

#
fig=plt.figure()
plt.subplot(121)
plt.pcolormesh(model.tstats[0, :, :], cmap='RdBu_r')
plt.colorbar()
#plt.plot((0, 375), (0, 375), 'k:')
plt.contour(clusters[0], 1)
plt.subplot(122)
plt.pcolormesh(clusters[0][:, :])

np.save('clst_p.npy',clusters[0])
fig.savefig()

#%% Max-Stats don't work so well for this, too many comparisons
nperms = 1000
P = MaxStatPermutation(design, data, 0, nperms, pooled_dims=(1, 2),
                       metric='tstats', nprocesses=6)
thresh = P.get_thresh([90])

plt.figure()
plt.hist(P.nulls, 32)
plt.hist(model.tstats.reshape(-1), 32)

plt.figure()
plt.subplot(121)
plt.pcolormesh(model.tstats[0, :, :], cmap='RdBu_r')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(model.tstats[0, :, :]>thresh)