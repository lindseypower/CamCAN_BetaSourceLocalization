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

# Average across 2 year age bins (35 bins in total)
min_ages = [18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86]

#List to store average stc for each group
stc_byGroup = []

for min_age in min_ages:
    print(min_age)
    max_age = min_age+1

    # Set folders and files
    dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/MEG1311/'
    subjectDir = '/home/timb/camcan/subjects/'
    stcPrefix = '/transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS_fsaverage'

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
    print(stcArray.shape)

    # Average over participants and make an stc
    stcGAvgData = np.mean(stcArray, axis=0)
    stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices, 
	tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
    
    #Add to list 
    stc_byGroup.append(stcGAvg)

#Loop through all the group averages and conduct t-tests between them
tstats = []
for x in stc_byGroup:
    tstat_row = []
    for y in stc_byGroup:
        ttest = ss.ttest_rel(x.data, y.data)
        stat = ttest.statistic[0]
        tstat_row.append(stat)
    tstats.append(tstat_row)

#print tstats as an array
tstats = np.asarray(tstats)
tstats = np.absolute(tstats)
print(tstats)

#Plot data as a colourmap
viridis = cm.get_cmap('viridis', 12)
newcolors = viridis(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
psm = ax.pcolormesh(tstats, cmap=newcmp, rasterized=True, vmin=0, vmax=400)
fig.colorbar(psm, ax=ax)
