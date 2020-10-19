import os
import sys
import numpy as np
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs, maxwell_filter, find_bad_channels_maxwell
import multiprocessing as mp
import logging

def emptyroom_preprocess(subjectID):

    #Filenames
    emptyRoomFif = '/home/timb/data/camcan/download/20170824/cc700/meg/pipeline/release004/emptyroom/' + subjectID + '/emptyroom_' + subjectID + '.fif'
    rawTaskFif = '/home/timb/camcan/megData/' + subjectID + '/task/task_raw.fif'
    calibration = '/home/timb/camcan/camcanMEGcalibrationFiles/sss_cal.dat'
    cross_talk = '/home/timb/camcan/camcanMEGcalibrationFiles/ct_sparse.fif'

    outDir = '/media/NAS/lpower/BetaSourceLocalization/emptyroomData/' + subjectID
    icaFif = outDir + '/emptyroom_trans-ica.fif'
    epochFif = outDir + '/emptyroom_trans-epo.fif'
    evokedFif = outDir + '/emptyroom_trans-ave.fif'
    eveFif_all = outDir + '/emptyroom_trans-eve.fif'

    #Read in files 
    raw = mne.io.read_raw_fif(rawTaskFif)
    raw_emptyroom = mne.io.read_raw_fif(emptyRoomFif, preload=True)

    #Set the head transformation info in the empty room data to match the task data 
    raw_emptyroom.info['dev_head_t'] = raw.info['dev_head_t']

    #Find bad channels and add them to info
    #The MNE documentation said to use 'meg' coordinate frame for empty-room noise and it crashes if I don't, but I'm not sure if this is going against the transformation that we've done
    raw_emptyroom.info['bads'] = []
    raw_check = raw_emptyroom.copy()
    auto_noisy_chs, auto_flat_chs  = find_bad_channels_maxwell(raw_check, cross_talk=cross_talk, calibration=calibration,verbose=True, coord_frame='meg')
    print(auto_noisy_chs)
    print(auto_flat_chs)

    bads = raw_emptyroom.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw_emptyroom.info['bads'] = bads

    #Maxfilter the empty room data
    raw_emptyroom = mne.preprocessing.maxwell_filter(raw_emptyroom, cross_talk=cross_talk, calibration=calibration,verbose=True, coord_frame='meg')

    #Filter raw data 
    raw_emptyroom.filter(l_freq=None, h_freq=125)
    raw_emptyroom.notch_filter([50,100])

    #There are no events in the raw data so need to make some - exact procedure used to process the raw data
    print(len(raw_emptyroom))
    evs = mne.make_fixed_length_events(raw_emptyroom, start=30.0, duration=30.0)
    print(evs)
    evs = evs[0,:]
    evs = evs.reshape([1,3])
    mne.write_events(eveFif_all, evs)

    # Epoch data based on button press
    epochs = mne.Epochs(raw_emptyroom, evs, None, -30.0, 30.0,verbose=False, preload=True)
    # Load or generate ICA decomposition for this dataset
    # performs ICA on data to remove artifacts according to rejection criteria
    if os.path.exists(icaFif):
        print('Reading ICA: ' + icaFif)
        ica = mne.preprocessing.read_ica(icaFif)
    else:
        print('Running ICA')
        reject = dict(grad=4000e-13, mag=5e-12)
        picks = mne.pick_types(raw_emptyroom.info, meg=True, eeg=False, eog=True, stim=False, exclude='bads')
        ica = ICA(n_components=0.99, method='fastica')
        ica.fit(raw_emptyroom, picks=picks, reject=None)
        n_max_ecg, n_max_eog = 3, 3
        
        # Reject bad EOG components following mne procedure
        try:
            eog_epochs = create_eog_epochs(raw_emptyroom, tmin=-0.5, tmax=0.5, reject=reject)
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
            eog_inds = eog_inds[:n_max_eog]
            ica.exclude.extend(eog_inds)
        except:
            print("""Subject {0} had no eog/eeg channels""".format(str(subjectID)))
        
        # Reject bad ECG compoments following mne procedure
        ecg_epochs = create_ecg_epochs(raw_emptyroom, tmin=-0.5, tmax=0.5)
        ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        ecg_inds = ecg_inds[:n_max_ecg]
        ica.exclude.extend(ecg_inds)
        
        # save ICA file
        ica.save(icaFif)

    # Apply ICA to epoched data and save
    epochs_clean = epochs.copy()
    ica.apply (epochs_clean, exclude=ica.exclude)
    epochs_clean.save(epochFif)
    
    # Average and save
    evoked = epochs_clean.average()
    evoked.save(evokedFif)
    
    print (str(subjectID))
    print (str(len(epochs)))
    print (str(len(ica.exclude)))

    return epochs_clean

epochs = emptyroom_preprocess('CC110033')
