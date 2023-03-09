def Cat_same(sub):

    

    participant_arr=['221128']
    result_all_path='/rds/projects/j/jenseno-opm/cross_modal_project/Across_participants/Category/'

    data_name = 'full'
    participant=participant_arr[sub-1]
    
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path=data_path + '/proccessed/'
    
    #data_path =r'Z:/cross_modal_project/221130/'
    #result_path=r'Z:/cross_modal_project/221130/proccessed/'
    
    
    path_file = os.path.join(result_path,data_name+'_supertrials.fif') 
    epochs_raw = mne.read_epochs(path_file, preload=True,verbose=True)
    epochs_raw.filter(1,30)
    epochs_raw.resample(300)
    epochs = epochs_raw.copy().crop(-0.1,0.7)
    epochs.apply_baseline(baseline=(-0.1, 0))
       
    Category = {
            "11":"move",
            "12":"still",
            "21":"big",
            "22":"small",
            "31":"nat",
            "32":"man"
    }
    Category_item ={
            0:"000",
            1:"001",
            2:"010",
            3:"011",
            4:"100",
            5:"101",
            6:"110",
            7:"111"
    }
    lngth=len(epochs['start_000/w/still/small/man'][0].get_data(picks='meg')[0,1,:])
    rsa_mat=np.zeros([48,48,round(lngth/2)+1])
    ii=0
    jj=0
    cat=[0,1,2,3,4,5,6,7]
    item=[0,1,2,3,4,5]
    time_epoch=len(epochs['start_000/w/still/small/man'][0].get_data(picks='meg')[0,1,:]
    pairs=list(itertools.product(cat,item))

    for count_t, tt in enumerate(range(0,time_epoch,2),0):        
        for it_pairs_w , ww in enumerate(pairs):
            for it_pairs_p , pp in enumerate(pairs)):

                a_w=epochs['p/'+'start_'+Category_item[ww[0]]][ww[1]].get_data(picks='meg')[0,:,:]
                a_p=epochs['p/'+'start_'+Category_item[pp[0]]][pp[1]].get_data(picks='meg')[0,:,:]
                r, p = scipy.stats.pearsonr(a_w[:,tt], a_p[:,tt])
                rsa_mat[it_pairs_w,it_pairs_p,count_t]=r

        print('time '+str(tt))
    np.save(result_path+'rsa_mat_pp',rsa_mat)
    #fig=plt.figure()
    #plt.matshow(rsa_mat[:,:,0])
    #colorbar()
    #corr_mat=np.zeros([48,121])
    #for it in range(48):
    #a    corr_mat[it,:]=rsa_mat[it,it,:]
    
if __name__ == "__main__":
    import sys
    from init_y import *
    Cat_same(int(sys.argv[1]))
