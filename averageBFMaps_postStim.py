import mne
import os
import numpy as np

# Script to read in DICS beamformer maps for transient beta events in the
#	prestimulus interval of CamCAN data
#
# Note: data files are copied from Biden folder:
#		/home/timb/camcan/spectralEvents

# Set folders and files
channelName = 'MEG1311'
dataDir = '/media/NAS/lpower/BetaSourceLocalization/postStimData/'+ channelName +'/'
subjectDir = '/home/timb/camcan/subjects/'
stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_postBetaEvents_dSPM_fsaverage'

#Find all subject folders that exist
subjects = os.listdir(dataDir)
print(subjects)      
# Loop over all subject folders
stcs = []
for subjectID in subjects:
     
    # Set file path for stc file (without Xh.stc)
    thisStcFile = os.path.join(dataDir, subjectID, stcPrefix)
     
    # Set file path with lh.stc to make sure the file exists
    fileCheckName = "".join([thisStcFile, '-lh.stc'])
     
    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs.append(stc.data)
     
# Turn list of stc data elements into an array (participants x vertices x 1)
print(len(stcs))
stcArray = np.asarray(stcs)
     
# Average over participants and make an stc
stcGAvgData = np.mean(stcArray, axis=0)
stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices,
             tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')# Find all subject folders that exist

outFileName = dataDir + 'postStim_dSPM_stcGAvg'
stcGAvg.save(outFileName)
print(str(len(stcArray)))
'''
# Plot the grand average stc
print('Plotting average BF map for ' + str(len(stcArray)) + ' subjects')
stcGAvg.plot(surface='pial', hemi='lh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True)
'''
