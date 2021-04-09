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

for subjectID in subjects: 

    #Localization parameters
    startTime = 0.25
    endTime = 1.25

    # Setup paths and names for file
    channelName = 'MEG1311'

    dataDir = '/home/timb/camcan/'
    MEGDir = os.path.join(dataDir, 'proc_data/TaskSensorAnalysis_transdef')
    outDir = os.path.join('/media/NAS/lpower/BetaSourceLocalization/postStimData',channelName, subjectID)
    subjectsDir = os.path.join(dataDir, 'subjects/')

    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFif = os.path.join(MEGDir, subjectID, epochFifFilename)
    
    #get post-movement timecourse 
    epochs = mne.read_epochs(epochFif)
    evoked = epochs.average()
    postMove = evoked.crop(tmin=startTime, tmax=endTime)

    
