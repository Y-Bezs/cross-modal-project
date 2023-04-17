def MEG_Perform_Classification(epoch1_train, epoch2_train, epoch1_test, epoch2_test, suj, filename, times, liminf, limsup):
    
    # filename = 'FixationOnsetEncoding_Color_vs_Grey
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                              cross_val_multiscore, LinearModel, get_coef,
                              Vectorizer, CSP)
    import sklearn.svm
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Testing
    # epoch1_train = EpochColorFovealClass
    # epoch2_train = EpochGreyFovealClass
    # epoch1_test = EpochColorClass
    # epoch2_test = EpochGreyClass
    # times = timesColor
    
    # %% Prepare training data
    # Matrix X (features) is a three-dimensional matrix (trials x channel x time points)
    X_train = np.vstack((epoch1_train, epoch2_train))
    # Vector Y (targets)
    nEpoch1_train = len(epoch1_train)
    nEpoch2_train = len(epoch2_train)
    iEpoch1_train = np.ones([1,nEpoch1_train])
    iEpoch2_train = 2*np.ones([1,nEpoch2_train])
    Y_train = np.append(iEpoch1_train, iEpoch2_train)
    
    # %% Prepare testing data
    # Matrix X (features) is a three-dimensional matrix (trials x channel x time points)
    X_test = np.vstack((epoch1_test, epoch2_test))
    # Vector Y (targets)
    nEpoch1_test = len(epoch1_test)
    nEpoch2_test = len(epoch2_test)
    iEpoch1_test = np.ones([1,nEpoch1_test])
    iEpoch2_test = 2*np.ones([1,nEpoch2_test])
    Y_test = np.append(iEpoch1_test, iEpoch2_test)
    
    # %% Classifier settings 
 
    clf = make_pipeline(Vectorizer(),StandardScaler(),  
                       LinearModel(sklearn.svm.SVC(kernel = 'linear')))     
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
    
    # %% Fit classifier
    
    time_decod.fit(X_train, Y_train)
    
    # %% Test classifier
    
    scores = time_decod.score(X_test, Y_test)
    
    # %% Fit and test on the same dataset
    
    # scores = cross_val_multiscore(time_decod, X, Y, cv=5, n_jobs=-1)
    
    # %% Save and Plot 
    Classif_Filename = 'Data/'+suj+'/'+suj+'_Classification_TrainFoveal_'+filename+'.npy'
    np.save(Classif_Filename, scores)
    # Plot results
    # scores = np.mean(scores, axis=0)
    fig, ax = plt.subplots()
    plt.ylim([liminf, limsup]) #([0.35, 0.65])
    ax.plot(times, scores, label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    plt.savefig('Figures/Classification/'+suj+'_Classification_TrainFoveal_'+filename+'.png')
    
    # %% Temporal Generalization settings
    
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring='roc_auc',
                                      verbose=True)
    
    # %% Fit classifier
    
    time_gen.fit(X_train, Y_train)
    
    # %% Test classifier
    
    scores_timegen = time_gen.score(X_test, Y_test)
    
    # %% Fit and test on the same dataset
    
    # scores_timegen = cross_val_multiscore(time_gen,X, Y, cv=5, n_jobs=-1)
    
    # %% Save and Plot
    
    Classif_tempgene_Filename = 'Data/'+suj+'/'+suj+'_TemporalGeneralization_TrainFoveal_'+filename+'.npy'
    np.save(Classif_tempgene_Filename, scores_timegen)
    # Plot Results
    # scores_timegen = np.mean(scores_timegen, axis=0)
    fig, ax = plt.subplots(1, 1)
    plt.imshow(scores_timegen, interpolation='nearest', origin='lower', cmap='RdBu_r',
                vmin=liminf, vmax=limsup) # vmin=0.35, vmax=0.65)
    ax.set_xlabel('Times Test (ms)')
    ax.set_ylabel('Times Train (ms)')
    ax.set_title('Time generalization (%s vs. %s)') 
    plt.axvline(0, color='k')
    plt.axhline(0, color='k')
    plt.colorbar()
    plt.savefig('Figures/Classification/'+suj+'_TemporalGeneralization_TrainFoveal_'+filename+'.png')
    
    return 
    