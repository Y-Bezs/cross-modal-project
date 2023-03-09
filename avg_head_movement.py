from init_y import *

participant_arr=list(Part_info.keys())
for ii in range(28):#Part_info:
    participant=participant_arr[ii]
    data_path ='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/'
    result_path='/rds/projects/j/jenseno-opm/cross_modal_project/'+participant+'/proccessed/'
    old=1 if Part_info[participant]<109 else 0

    data_name = 'head_movementhead_locs_.npy'
    path_file = os.path.join(data_path, data_name)
    head_locs = np.load(path_file)