import mne 
import numpy as np
import os
import scipy.stats as ss

# Set folders and files (have to change these based on channel and condition)
channelName = 'MEG0221'
subjectDir = '/home/timb/camcan/subjects/'

#Pre-stim condition
pre_dataDir = '/media/NAS/lpower/BetaSourceLocalization/preStimData/'+ channelName
pre_stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preBetaEvents_dSPM_fsaverage-lh.stc'

#Rest condition
rest_dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'+ channelName 
rest_stcPrefix = 'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_dSPM_fsaverage-lh.stc'

#Find all subject folders that exist
subjects = os.listdir(pre_dataDir)
    
# Loop over all subject folders
diffs = []
sub_count = 0;
for subjectID in subjects:
     
    # Set file path for stc file
    preStcFile = os.path.join(pre_dataDir, subjectID, pre_stcPrefix)
    restStcFile = os.path.join(rest_dataDir, subjectID, rest_stcPrefix)

    # If files exist read in premovement and rest source estimates 
    if os.path.exists(preStcFile) and os.path.exists(restStcFile):
        pre_stc = mne.read_source_estimate(preStcFile)
        rest_stc = mne.read_source_estimate(restStcFile)
        
        #Take the difference between estimates (rest - pre-move)
        diff_stc = rest_stc.__sub__(pre_stc)
        #save source estimate data for this subject to a list
        diff_vertex_vals = diff_stc.data
        diffs.append(diff_vertex_vals)
        print(sub_count)
        sub_count = sub_count + 1

#reformat difference arrays for computing t-tests for each vertex
diffs = np.asarray(diffs)
diffs = np.reshape(diffs, (sub_count,20484))

#Perform t-test for each difference vertex across participants (null hypothesis: diff=0)
ttests = []
pvals = []
tstats = []
for i in range(0,diffs.shape[1]):
    thisTest = ss.ttest_1samp(diffs[:,i],0)
    ttests.append(thisTest)

    thispval = thisTest.pvalue
    pvals.append(thispval)

    thisTstat = thisTest.statistic
    tstats.append(thisTstat)

#Restructure pvals and create a source estimate object with the p-val data so it can be plotted 
pvals = np.asarray(pvals)
pvals = np.reshape(pvals, (20484,1))
pval_stc = diff_stc.copy()
pval_stc.data = pvals

#Do the same thing with the t-stat 
tstats = np.asarray(tstats)
tstats = np.reshape(tstats, (20484,1))
tstats_stc = diff_stc.copy()
tstats_stc.data = tstats

#Save and plot
outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/dSPM_pre_rest_pvals'
pval_stc.save(outFileName)

outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/dSPM_pre_rest_tstats'
tstats_stc.save(outFileName)

#pval_stc.plot(surface='pial', hemi='both', subjects_dir=subjectDir, subject='fsaverage',
 #       backend='mayavi', time_viewer=True, clim=clim)

