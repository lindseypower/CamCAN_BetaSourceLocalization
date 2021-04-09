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
import scipy.signal as ss

def calc_epochData(channelName): 
    #set folders and files 
    dataDir = '/media/NAS/lpower/BetaSourceLocalization/postStimData/'+ channelName +'/' 
    subjects = os.listdir(dataDir) 
    substring = 'CC' 
    epochsList = [] 
       
    for subjectID in subjects:  
        
        if substring in subjectID: 
            #Localization parameters 
            startTime = 0.25
            endTime = 1.25
            
            # Setup paths and names for file 
            dataDir = '/home/timb/camcan/' 
            MEGDir = os.path.join(dataDir, 'proc_data/TaskSensorAnalysis_transdef') 
            
            epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif' 
            epochFif = os.path.join(MEGDir, subjectID, epochFifFilename) 
            
            #get post-movement timecourse  
            epochs = mne.read_epochs(epochFif)
            postMove = epochs.crop(tmin=startTime, tmax=endTime)
            postMove = postMove.pick_channels([channelName])
            postMove_flat = postMove.get_data().flatten()
            epochsList.append(postMove_flat) 
            
    epochsArray = np.concatenate(epochsList) 
    
    return epochsArray

leftData = calc_epochData('MEG0221')
rightData = calc_epochData('MEG1311') 

f, Cxy = ss.coherence(leftData, rightData, fs=1000, nperseg=250, noverlap=0)
plt.plot(f, Cxy)
plt.xlim(0,100)

