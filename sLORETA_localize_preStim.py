# Import libraries
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.minimum_norm import make_inverse_operator, apply_inverse, source_band_induced_power
from mne.time_frequency import tfr_morlet
import multiprocessing as mp
import time

def make_sLORETA(subjectID):

    #Localization parameters 
    fmin = 15
    fmax = 30
    startTime = -1.25
    endTime = -0.25
    tmins = [0.0, -0.575]

    # Setup paths and names for file
    channelName = 'MEG0221'

    dataDir = '/home/timb/camcan/'
    MEGDir = os.path.join(dataDir, 'proc_data/TaskSensorAnalysis_transdef')
    outDir = os.path.join('/media/NAS/lpower/BetaSourceLocalization/preStimData',channelName, subjectID)
    subjectsDir = os.path.join(dataDir, 'subjects/')
     
    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFif = os.path.join(MEGDir, subjectID, epochFifFilename)
     
    spectralEventsCSV = 'MEG0221_spectral_events_-1.0to1.0s.csv'
    csvFile = '/media/NAS/lpower/camcan/spectralEvents/task/'+  channelName + '/'+ subjectID + '/' + spectralEventsCSV
     
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    
    emptyroomFif = '/media/NAS/lpower/BetaSourceLocalization/emptyroomData/' + subjectID + '/emptyroom_trans-epo.fif' 
     
    # Files to make
    stcFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3e.4s_cleaned-epo_preBetaEvents_sLORETA')
    stcMorphFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_sLORETA_fsaverage')
    testCompleteFile = os.path.join(outDir,'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_sLORETA-lh.stc')
    #if os.path.exists(testCompleteFile):
     #   return

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
        if (epochCrop.tmin == -0.575 and epochCrop.tmax == 0.575):
            epochList.append(epochCrop)

    epochs = mne.concatenate_epochs(epochList)
    epochs.pick_types(meg=True)
    
    #Need to read in the empty room noise to compute the noise covariance matrix with the empty room noise. 
    #Read in raw room noise data
    empty_room = mne.read_epochs(emptyroomFif)
    
    #Compute noise covariance
    noise_cov = mne.compute_covariance(empty_room, tmin=0, tmax=None)
    
    #Calculate evoked response (only used Mags for the DICS beamformer so can just use Mags for this)
    evoked = epochs.average().pick('meg')
    
    # Read source space
    src = mne.read_source_spaces(srcFif)
    # Make forward solution
    forward = mne.make_forward_solution(epochs.info,trans=transFif, src=src, bem=bemFif, meg=True, eeg=False)
                                        
    #Make MEG inverse operator 
    inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov)
    
    #Calculate the inverse solution using the MNE method
    method = "sLORETA"

    # Compute a source estimate per frequency band
    bands = dict(beta=[15,30])

    stc = source_band_induced_power(epochs, inverse_operator, bands, method=method, n_cycles=2,
                                 use_fft=False, n_jobs=1)

    baselineData = stc['beta'].data[:,0:400]
    activeData = stc['beta'].data[:,575:975]

    ERS = np.log2(activeData/baselineData)
    ERSstc = mne.SourceEstimate(ERS, vertices=stc['beta'].vertices, tmin=stc['beta'].tmin, tstep=stc['beta'].tstep, subject=stc['beta'].subject)
    
    ERSband = ERSstc.mean()
    ERSband.save(stcFile)

    #Compute morph file and save
    morph = mne.compute_source_morph(ERSband, subject_from='sub-' + subjectID,
                                     subject_to='fsaverage',
                                     subjects_dir=subjectsDir)
    ERSmorph = morph.apply(ERSband)
    ERSmorph.save(stcMorphFile)
    #print(stcMorphFile)
    return stc
    
if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = '/media/NAS/lpower/camcan/'
    dataDir = homeDir + 'spectralEvents/task/MEG0221'
    camcanCSV = dataDir + '/spectralEventAnalysis.csv'
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

    ex_subs = ['CC520395','CC222326','CC310414','CC320568', 'CC320636', 'CC321595', 'CC510534','CC520136','CC520745', 'CC520775', 'CC621080', 'CC720304']
    for x in ex_subs:
        subjectIDs.remove(x)

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/4))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(make_sLORETA, subjectIDs)
    #stc = make_dSPM('CC110033')
    
