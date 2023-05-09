# import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os.path as op
from init_y import *

# %% read data

cond = 'no_ica'
#data_path =r'Y:/cross_modal_project/Across_participants/Category_w_time/'
data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/Across_participants/Category_w_time/'
participant_arr=list(Part_info.keys())[3:]

suffics = { 1:'no_max',
            2:'max_wo_head',
            3:'max_w_head'}

xx=2  # choose suffics
sensor ='grad'
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

# %% Average all three categories for all participants, for pictures

sensor ='mag'
score_mean_pic = np.zeros((3,len(participant_arr), len(time)))
for xx in [1,2,3]: # all the filter conditions
    scores_all = scores_all = np.zeros((len(participant_arr),len(['w', 'p']) * len([11, 21, 31]), len(time)))
    file_name2='_'+suffics[xx]+'_'+ sensor+'_'+ cond + time_in
    for ii, par_name in enumerate(participant_arr):
        par=str(Part_info[par_name])
        file_name = file_name1 + par + file_name2 +'.npy'
        path_file = os.path.join(data_path+suffics[xx]+ '/' + cond +'/'+ file_name)
        scores = np.load(path_file)
        scores_all[ii,:,:] = scores # scores_all contains scores from all participants and has first three rows for words, lasr three for pictures.  Categories is in order of Category variable         
    score_mean_pic[xx-1,:,:] = np.mean(scores_all[:,3:6,:], axis = 1)

# %%   AUC paired t-test

start_ind = np.where(time==0.100)[0][0]
end_ind   = np.where(time==0.500)[0][0]
score_auc_pic = np.trapz(score_mean_pic[:,:,start_ind:end_ind],axis = 2)

t,p = stats.ttest_rel(score_auc_pic[0,:],score_auc_pic[2,:],axis=0,alternative = 'greater')

df = len(participant_arr)+len(participant_arr)-2
print('t(%g) = %g, p=%g'%(df,t,p))




# In[ ]:


## generate the data

# parameters
n1 = 30   # samples in dataset 1
n2 = 40   # ...and 2
mu1 = 1   # population mean in dataset 1
mu2 = 1.2 # population mean in dataset 2


# generate the data
data1 = mu1 + np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

# show their histograms
plt.hist(data1,bins='fd',color=[1,0,0,.5],label='Data 1')
plt.hist(data2,bins='fd',color=[0,0,1,.5],label='Data 2')
plt.xlabel('Data value')
plt.ylabel('Count')
plt.legend()
plt.show()


# In[ ]:


## now for the t-test

t,p = stats.ttest_ind(data1,data2,equal_var=True)

df = n1+n2-2
print('t(%g) = %g, p=%g'%(df,t,p))


# In[ ]:


## a 2D space of t values

# ranges for t-value parameters
meandiffs = np.linspace(-3,3,80)
pooledvar = np.linspace(.5,4,100)

# group sample size
n1 = 40
n2 = 30

# initialize output matrix
allTvals = np.zeros((len(meandiffs),len(pooledvar)))

# loop over the parameters...
for meani in range(len(meandiffs)):
    for vari in range(len(pooledvar)):
        
        # t-value denominator
        df = n1 + n2 - 2
        s  = np.sqrt(( (n1-1)*pooledvar[vari] + (n2-1)*pooledvar[vari]) / df)
        t_den = s * np.sqrt(1/n1 + 1/n2)
        
        # t-value in the matrix
        allTvals[meani,vari] = meandiffs[meani] / t_den

        
plt.imshow(allTvals,vmin=-4,vmax=4,extent=[pooledvar[0],pooledvar[-1],meandiffs[0],meandiffs[-1]],aspect='auto')
plt.xlabel('Variance')
plt.ylabel('Mean differences')
plt.colorbar()
plt.title('t-values as a function of difference and variance')
plt.show()