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
min_age = 47
max_age = 60
filename ='47-60_dSPM_stcGavg'

# Set folders and files
dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/MEG0221/'
subjectDir = '/home/timb/camcan/subjects/'
stcPrefix = '/transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_dSPM_fsaverage'

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
