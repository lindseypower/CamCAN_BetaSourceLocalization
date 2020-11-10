import mne 
import numpy as np
import os
import scipy.stats as ss

# Set folders and files (have to change these based on channel and condition)
channelName = 'MEG0221'
subjectDir = '/home/timb/camcan/subjects/'

#Rest condition
rest_dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'+ channelName 
rest_stcPrefix = 'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_dSPM_fsaverage-lh.stc'

#Post-stim condition
post_dataDir = '/media/NAS/lpower/BetaSourceLocalization/postStimData/'+ channelName
post_stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_postBetaEvents_dSPM_fsaverage-lh.stc'

#Find all subject folders that exist
subjects = os.listdir(post_dataDir)
    
# Loop over all subject folders
diffs = []
sub_count = 0;
for subjectID in subjects:
     
    # Set file path for stc file
    restStcFile = os.path.join(rest_dataDir, subjectID, rest_stcPrefix)
    postStcFile = os.path.join(post_dataDir, subjectID, post_stcPrefix)

    # If files exist read in premovement and postmove source estimates 
    if os.path.exists(restStcFile) and os.path.exists(postStcFile):
        rest_stc = mne.read_source_estimate(restStcFile)
        post_stc = mne.read_source_estimate(postStcFile)
        
        #Take the difference between estimates (post - pre-move)
        diff_stc = post_stc.__sub__(rest_stc)
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
outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/dSPM_rest_post_pvals'
pval_stc.save(outFileName)

outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/dSPM_rest_post_tstats'
tstats_stc.save(outFileName)

#pval_stc.plot(surface='pial', hemi='both', subjects_dir=subjectDir, subject='fsaverage',
 #       backend='mayavi', time_viewer=True, clim=clim)

