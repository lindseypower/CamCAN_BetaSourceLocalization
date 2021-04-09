import mne
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# Script to read in DICS beamformer maps for transient beta events in the
#	prestimulus interval of CamCAN data
#
# Note: data files are copied from Biden folder:
#		/home/timb/camcan/spectralEvents

#List to store average stc for each group
stc_byGroup = []

datDir = '/media/NAS/lpower/camcan/'
camcanCSV = datDir + 'oneCSVToRuleThemAll.csv'
subjectData = pd.read_csv(camcanCSV)

gender1 = subjectData[subjectData['Gender_x'] == 1]
gender2 = subjectData[subjectData['Gender_x'] == 2]
gender1_subs = gender1['SubjectID'].tolist()
gender2_subs = gender2['SubjectID'].tolist()

# Set folders and files
dataDir = '/media/NAS/lpower/BetaSourceLocalization/preStimData/MEG0221/'
subjectDir = '/home/timb/camcan/subjects/'
stcPrefix = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_DICS_fsaverage'

# Loop over all subject folders
stcs = []
for subjectID in gender1_subs:
    # Set file path for stc file (without Xh.stc)
    thisStcFile = dataDir + subjectID + stcPrefix

    # Set file path with lh.stc to make sure the file exist
    fileCheckName = "".join([thisStcFile, '-lh.stc'])
        
    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs.append(stc.data)

stcs2 = []
for subjectID in gender2_subs:
    # Set file path for stc file (without Xh.stc)
    thisStcFile = dataDir + subjectID + stcPrefix
     
    # Set file path with lh.stc to make sure the file exist
    fileCheckName = "".join([thisStcFile, '-lh.stc'])
             
    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs2.append(stc.data)

# Turn list of stc data elements into an array (participants x vertices x 1)
stcArray = np.asarray(stcs)
stcArray2 = np.asarray(stcs2)

# Average over participants and make an stc
stcGAvgData = np.mean(stcArray, axis=0)
stcGAvgData2 = np.mean(stcArray2, axis=0)
stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices, 
    tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
stcGAvg2 = mne.SourceEstimate(stcGAvgData2, vertices=stc.vertices,
    tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')

#conduct ttest to compare male and female averages
ttest = ss.ttest_rel(stcGAvg.data, stcGAvg2.data)
ttest.statistic
'''
outFileName = dataDir + 'rest_dSPM_stcGAvg_gender1'
stcGAvg.save(outFileName)

outFileName = dataDir + 'rest_dSPM_stcGAvg_gender2'
stcGAvg2.save(outFileName)
'''
