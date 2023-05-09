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
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import List, Optional
from scipy.spatial.distance import squareform
#import rsatoolbox

Part_info={
        '221107':105, #0
        '221110':107, #1
        '221114':108, #2
        '221121':111, #3
        '221124':112, #4
        '221125':113, #5
        '221128':114, #6
        '221129':115, #7
        '221130':116, #8
        '221213':117, #9
        '230113':118, #10
        '230118':119,
        '230119':120,
        '230120':121,
        '230124':122,
        #'230125':123, 
        #'230126':124,
        '230127':125, #15
        #'230130':126, #too much wrong answers
        '230131':127,
        #'230202':128, #too much wrong answers, sleeping
        '230206':129,
        '230207':130, #18
        '230208':131,
        '230209':132, #20
        #'230214':133, 
        '230215':134,
        #'230217':135,
        '230216':136, #22
        '230222':137,
        '230223':138, 
        '230224':139, #25
        '230227':140,
        '230302':141, #27
        '230303':142,
        '230306':143,
        #'230308':144,
        '230309':145 #34

}

data_path ='/rds/projects/k/kowalcau-opm-recordings/cross_modal_project/'
data_name = 'full'

events_id = {
        'start_000/w/still/small/man':240+1,
        'start_100/w/move/small/man':32+1, 
        'start_010/w/still/big/man':64+1, 
        'start_110/w/move/big/man':96+1, 
        'start_001/w/still/small/nat':128+1, 
        'start_101/w/move/small/nat':160+1,
        'start_011/w/still/big/nat':192+1, 
        'start_111/w/move/big/nat':224+1,
        'start_000/p/still/small/man':240+2,
        'start_100/p/move/small/man':32+2, 
        'start_010/p/still/big/man':64+2, 
        'start_110/p/move/big/man':96+2,
        'start_001/p/still/small/nat':128+2, 
        'start_101/p/move/small/nat':160+2, 
        'start_011/p/still/big/nat':192+2,
        'start_111/p/move/big/nat':224+2
        }



