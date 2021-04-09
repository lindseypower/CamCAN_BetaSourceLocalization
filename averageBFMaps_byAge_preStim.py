import mne
import os
import numpy as np
import pandas as pd

# Script to read in DICS beamformer maps for transient beta events in the
#	prestimulus interval of CamCAN data
#
# Note: data files are copied from Biden folder:
#		/home/timb/camcan/spectralEvents

# Some variables 
min_age = 18
max_age = 32
filename = '18-32_DICS_stcGavg'

# Set folders and files
dataDir = '/media/NAS/lpower/BetaSourceLocalization/preStimData/MEG1311/'
subjectDir = '/home/timb/camcan/subjects/'
stcPrefix = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_DICS_fsaverage'

# Find subjects to be analysed
datDir = '/media/NAS/lpower/camcan/'
camcanCSV = datDir + 'oneCSVToRuleThemAll.csv'
subjectData = pd.read_csv(camcanCSV)
     
# Take only subjects with more than 55 epochs
subjectData = subjectData[subjectData['Age_x'] >= min_age]
subjectData = subjectData[subjectData['Age_x'] <= max_age]
subjectIDs = subjectData['SubjectID'].tolist()

# Loop over all subject folders
stcs = []
for subjectID in subjectIDs:
    # Set file path for stc file (without Xh.stc)
    thisStcFile = dataDir + subjectID + stcPrefix

    # Set file path with lh.stc to make sure the file exist
    fileCheckName = "".join([thisStcFile, '-lh.stc'])
        
    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs.append(stc.data)

# Turn list of stc data elements into an array (participants x vertices x 1)
stcArray = np.asarray(stcs)

# Average over participants and make an stc
stcGAvgData = np.mean(stcArray, axis=0)
stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices, 
	tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
outFileName = dataDir + filename
stcGAvg.save(outFileName)
print(str(len(stcArray)))
'''
# Plot the grand average stc
print('Plotting average BF map for ' + str(len(stcArray)) + ' subjects')
stcGAvg.plot(surface='pial', hemi='lh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True)
'''        
