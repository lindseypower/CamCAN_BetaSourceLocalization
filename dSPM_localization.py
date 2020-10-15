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

def make_MNE(subjectID):

    # Setup paths and names for file
    dataDir = '/home/timb/camcan/'
    MEGDir = os.path.join(dataDir, 'proc_data/TaskSensorAnalysis_transdef')
    outDir = os.path.join('/media/NAS/lpower/camcan/BetaSourceLocalization/preStimData',channelName, subjectID)
    subjectsDir = os.path.join(dataDir, 'subjects/')
     
    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFif = os.path.join(MEGDir, subjectID, epochFifFilename)
     
    spectralEventsCSV = "".join([subjectID, 'MEG0221_spectral_events_-1.0to1.0s.csv'])
    csvFile = os.path.join('/media/NAS/lpower/camcan/spectralEvents/task/', subjectID, spectralEventsCSV)
     
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    
    emptyRoomFif =
     
    # Files to make
    stcFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_MNE')
    stcMorphFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_MNE_fsaverage')
    testCompleteFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_MNE-lh.stc')
    if os.path.exists(testCompleteFile):
        return

    else:
        if not os.path.exists(outDir):
            os.makedirs(outDir)
            
    # Read all transient events for subject
    df = pd.read_csv(csvFile)
    # Events that meet Shin criteria only
    df1 = df[df['Outlier Event']]
    # Freq range of interest
    df2 = df1.drop(df1[df1['Peak Frequency'] < fmin].index)
    df3 = df2.drop(df2[df2['Peak Frequency'] > fmax].index)
    df4 = df3.drop(df3[df3['Peak Time'] > endTime].index)
    newDf = df4.drop(df4[df4['Peak Time'] < startTime].index)
    
    #I only want to take the top 55 highest power beta events (based on pre-analysis calculations in R)
    #If the dataframe has less than 55 values, return 
    newDf = newDf.sort_values(by='Normalized Peak Power')
    if newDf.size >= 55:
        newDf = newDf.tail(n=55)
    else: 
        print("Not enough bursts to make a map.")
        return

    # Read epochs
    originalEpochs = mne.read_epochs(epochFif)

    # Re-calculate epochs to have one per spectral event - this is the part that ensures that the map is calculated per burst
    ## IMPORTANT FOR METHODS DESCRIPTION ##
    numEvents = len(newDf)
    epochList = []
    for e in np.arange(numEvents):
        thisDf = newDf.iloc[e]
        onsetTime = thisDf['Event Onset Time']
        epoch = originalEpochs[thisDf['Trial']]
        epochCrop = epoch.crop(onsetTime+tmins[1], onsetTime-tmins[1])
        epochCrop = epochCrop.apply_baseline(baseline=(None,None))
        # Fix epochCrops times array to be the same every time = (-.4, .4)
        epochCrop.shift_time(tmins[1], relative=False)
        epochList.append(epochCrop)

    epochs = mne.concatenate_epochs(epochList)
    epochs.pick_types(meg=True)
    
    #Need to read in the empty room noise to compute the noise covariance matrix with the empty room noise. 
    #Read in raw room noise data
    raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname)
    
    #FILTER THIS THE SAME WAY OTHER DATA WAS FILTERED
    
    #Compute noise covariance
    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
    
    #Calculate evoked response (only used Mags for the DICS beamformer so can just use Mags for this)
    evoked = epochs.average().pick('meg')
    
    # Read source space
    src = mne.read_source_spaces(srcFif)
    # Make forward solution
    forward = mne.make_forward_solution(epochs.info,trans=transFif, src=src, bem=bemFif, meg=True, eeg=False)
                                        
    #Make MEG inverse operator 
    inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov)
    
    #Calculate the inverse solution using the MNE method
    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc, residual = apply_inverse(evoked, inverse_operator, lambda2,method=method, pick_ori=None,return_residual=True, verbose=True)
    
    #Save stc file 
    stc.save(stcFile)
    
    #Compute morph file and save
    morph = mne.compute_source_morph(stc, subject_from='sub-' + subjectID,
                                     subject_to='fsaverage',
                                     subjects_dir=subjectsDir)
    morph.save(stcMorphFile)
    
    return
    
if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'spectralEvents/spectralEventAnalysis.csv'
    subjectData = pd.read_csv(camcanCSV)

    # Take only subjects with more than 55 epochs
    subjectData = subjectData[subjectData['numEpochs'] > 55]

    # Drop subjects with PMBR stc already made
    subjectData = subjectData.drop(subjectData[subjectData['bemExists'] == False].index)
    subjectData = subjectData.drop(subjectData[subjectData['srcExists'] == False].index)
    subjectData = subjectData.drop(subjectData[subjectData['transExists'] == False].index)
    subjectData = subjectData.drop(subjectData[subjectData['PreStim Stc Exists'] == True].index)

    subjectIDs = subjectData['SubjectID'].tolist()
    print(len(subjectIDs))
    print(subjectIDs)

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/4))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(make_MNE, subjectIDs)

    