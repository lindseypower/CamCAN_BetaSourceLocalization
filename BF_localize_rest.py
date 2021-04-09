#!/usr/bin/env python`

# Import libraries
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import tfr_morlet
import multiprocessing as mp
import time

mne.set_log_level('DEBUG')

# Mapping the transient beta bursts that occur during the rebound interval

def make_BF_map(subjectID):
    """Top-level run script for making spectral events BF map from MEG data."""
    print(subjectID)
    #################################
    # Variables

    channelName = 'MEG1311'

    # Analysis Paramaters for Picking Transient Events      
    fmin = 15               # NOTE: no constraints on time for the resting data (across entire time interval)
    fmax = 30

    # TFR analysis parameters
    TFRfmin = 5
    TFRfmax = 60
    TFRfstep = 5
    
    # DICS Settings
    tmins = [0.0, -0.575] # Start of each window (active, baseline) relative to event onset time
    tstep = 0.4         # Duration of BF windows in seconds, based on distribution of event durations 
    numFreqBins = 10  # linear spacing
    DICS_regularizaion = 0.5
    data_decimation = 1
    
    plotOK = False

    ###############
    # Setup paths and names for file
    dataDir = '/home/timb/camcan/'
    MEGDir = os.path.join('/media/NAS/lpower/camcan/spectralEvents/rest/proc_data')
    outDir = os.path.join('/media/NAS/lpower/BetaSourceLocalization/restData', channelName,  subjectID)
    subjectsDir = os.path.join(dataDir, 'subjects/')

    epochFifFilename = 'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo.fif'
    epochFif = os.path.join(MEGDir, subjectID, epochFifFilename)

    spectralEventsCSV = subjectID + '_MEG1311_spectral_events.csv'
    csvFile = os.path.join('/media/NAS/bbrady/random old results/spectralEventsRest/', subjectID,  spectralEventsCSV)

    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'

    emptyroomFif = '/media/NAS/lpower/BetaSourceLocalization/emptyroomData/' + subjectID + '/emptyroom_trans-epo.fif' #Added this file **NEW**

    # Files to make
    stcFile = os.path.join(outDir,
                           'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS')
    stcMorphFile = os.path.join(outDir,
                                'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS_fsaverage')
    testCompleteFile = os.path.join(outDir,
                           'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS-lh.stc')

    if os.path.exists(testCompleteFile):

        return

    else:
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        #####################################
        # Pull events from CSV

        # Read all transient events for subject
        df = pd.read_csv(csvFile)
        # Events that meet Shin criteria only
        df1 = df[df['Outlier Event']]
        # Freq range of interest
        df2 = df1.drop(df1[df1['Lower Frequency Bound'] < fmin].index)
        newDf  = df2.drop(df2[df2['Upper Frequency Bound'] > fmax].index)

        newDf = newDf.sort_values(by='Normalized Peak Power')
        if newDf.size >= 55:
            newDf = newDf.tail(n=55)
        else:
            print("Not enough bursts to create a map.")
            return

        if plotOK:
            # Raster plot of event onset and offset times
            ax = sns.scatterplot(x='Event Onset Time', y='Trial', data=newDf)
            sns.scatterplot(x='Event Offset Time', y='Trial', data=newDf, ax=ax)
            plt.show()

            # Distribution of event durations
            sns.distplot(newDf['Event Duration'])
            plt.show()
            # Based on the distribution, an interval of 0-400 ms will include the full event duration in most cases

        ##############################################
        # Now do the DICS beamformer map calcaulation

        # Read epochs
        originalEpochs = mne.read_epochs(epochFif)

        # Re-calculate epochs to have one per spectral event
        numEvents = len(newDf)
        #print(str(numEvents) + ' events')
        epochList = []
        for e in np.arange(numEvents):
            thisDf = newDf.iloc[e]
            onsetTime = thisDf['Event Onset Time']
            epoch = originalEpochs[thisDf['Trial']]
            epochCrop = epoch.crop(onsetTime+tmins[1], onsetTime+tstep)
            epochCrop = epochCrop.apply_baseline(baseline=(None,None))
            # Fix epochCrops times array to be the same every time = (-.4, .4)
            epochCrop.shift_time(tmins[1], relative=False)
            if (epochCrop.tmax == tstep):
                epochList.append(epochCrop)
 
        epochs = mne.concatenate_epochs(epochList)
        epochs.pick_types(meg=True)

        '''
        # Let's look at the TFR across sensors
        magPicks = mne.pick_types(epochs.info, meg='mag', eeg=False, eog=False, stim=False, exclude='bads')
        freqs = np.arange(TFRfmin, TFRfmax, TFRfstep)
        n_cycles = freqs / 2.0
        power, _ = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, picks=magPicks,
                              use_fft=False, return_itc=True, decim=1, n_jobs=1)
        #power.save(tfrFile, overwrite=True)
        if plotOK:
            power.plot_joint(baseline=(-0.4, 0), mode='mean',
                             timefreqs = [(.2, 20)])
        '''

        # Read source space
        src = mne.read_source_spaces(srcFif)
        # Make forward solution
        forward = mne.make_forward_solution(epochs.info,
                                            trans=transFif, src=src, bem=bemFif,
                                            meg=True, eeg=False)

        # DICS Source Power example
        # https://martinos.org/mne/stable/auto_examples/inverse/plot_dics_source_power.html#sphx-glr-auto-examples-inverse-plot-dics-source-power-py

        #Compute noise csd from empty room data **NEW**
        epochs_emptyroom = mne.read_epochs(emptyroomFif)
        epochs_emptyroomMAG = epochs_emptyroom.pick_types(meg='mag')
        csd_emptyroom = csd_morlet(epochs_emptyroomMAG, decim=data_decimation, frequencies= np.linspace(fmin, fmax, numFreqBins))

        # Compute DICS spatial filter and estimate source power.
        stcs = []
        epochsMAG = epochs.copy()
        epochsMAG.pick_types(meg='mag')
        for tmin in tmins:
            csd = csd_morlet(epochsMAG, tmin=tmin, tmax=tmin + tstep, decim=data_decimation,
                             frequencies=np.linspace(fmin, fmax, numFreqBins))
            filters = make_dics(epochsMAG.info, forward, csd, noise_csd=csd_emptyroom, reg=DICS_regularizaion)
            stc, freqs = apply_dics_csd(csd, filters)
            stcs.append(stc)

        # Take difference between active and baseline, and mean across frequencies
        ERS = np.log2(stcs[0].data / stcs[1].data)
        a = stcs[0]
        ERSstc = mne.SourceEstimate(ERS, vertices=a.vertices, tmin=a.tmin, tstep=a.tstep, subject=a.subject)
        ERSband = ERSstc.mean()
        ERSband.save(stcFile)

        #ERSmorph = ERSband.morph(subject_to='fsaverage', subject_from='sub-' + subjectID, subjects_dir=subjectsDir)
        morph = mne.compute_source_morph(ERSband, subject_from='sub-' + subjectID,
                                         subject_to='fsaverage',
                                         subjects_dir=subjectsDir)
        ERSmorph = morph.apply(ERSband)
        ERSmorph.save(stcMorphFile)
        
        print(subjectID)
        return

if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = '/media/NAS/lpower'
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'spectralEvents/rest/events_data/MEG0221/spectralEventAnalysis.csv'
    subjectData = pd.read_csv(camcanCSV)

    # Take only subjects with more than 55 epochs
#    subjectData = subjectData[subjectData['numEpochs'] > 55]

    # Drop subjects with MR files missing
    subjectData = subjectData.drop(subjectData[subjectData['bemExists'] == False].index)
    subjectData = subjectData.drop(subjectData[subjectData['srcExists'] == False].index)
    subjectData = subjectData.drop(subjectData[subjectData['transExists'] == False].index)
    #subjectData = subjectData.drop(subjectData[subjectData['Stc Exists'] == True].index)

    subjectIDs = subjectData['SubjectID'].tolist()

    ex_subs = ['CC520395','CC222326','CC310414','CC320568', 'CC320636', 'CC321595', 'CC510534','CC520136','CC520745', 'CC520775', 'CC621080', 'CC720304']
    for x in ex_subs:
        subjectIDs.remove(x)

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/4))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(make_BF_map, subjectIDs)

    # Or run one subject for testing purposes
    #make_BF_map('CC110033')

