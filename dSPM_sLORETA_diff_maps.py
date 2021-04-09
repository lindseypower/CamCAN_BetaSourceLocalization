import mne 
import numpy as np
import os
import scipy.stats as ss

# Set folders and files (have to change these based on channel and condition)
channelName = 'MEG0221'
subjectDir = '/home/timb/camcan/subjects/'

#dSPM postStim data
dataDir = '/media/NAS/lpower/BetaSourceLocalization/postStimData/'+ channelName
dSPM_stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_postBetaEvents_dSPM_fsaverage-lh.stc'

#sLORETA postStim data
sLORETA_stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_postBetaEvents_sLORETA_fsaverage-lh.stc'

#Find all subject folders that exist
subjects = ['CC110033','CC210182','CC310086','CC420236','CC521040']
    
# Loop over all subject folders
diffs = []
sub_count = 0;
for subjectID in subjects:
     
    # Set file path for stc file
    dSPMStcFile = os.path.join(dataDir, subjectID, dSPM_stcPrefix)
    sLORETAStcFile = os.path.join(dataDir, subjectID, sLORETA_stcPrefix)

    # If files exist read in premovement and rest source estimates 
    #if os.path.exists(dSPMStcFile) and os.path.exists(sLORETAStcFile):
    print(dSPMStcFile)
    dSPM_stc = mne.read_source_estimate(dSPMStcFile)
    sLORETA_stc = mne.read_source_estimate(sLORETAStcFile)
        
    #Take the difference between estimates (dSPM - sLORETA)
    diff_stc = dSPM_stc.__sub__(sLORETA_stc)
    #save source estimate data 
    diffFile = os.path.join(dataDir, subjectID, 'dSPM-sLORETA_diff')
    print(diffFile)
    diff_stc.save(diffFile)
'''
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
#Save and plot
outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/sLORETA_pre_rest_pvals'
pval_stc.save(outFileName)

outFileName = '/media/NAS/lpower/BetaSourceLocalization/comparisonMaps/sLORETA_pre_rest_tstats'
tstats_stc.save(outFileName)
'''
#pval_stc.plot(surface='pial', hemi='both', subjects_dir=subjectDir, subject='fsaverage',
 #       backend='mayavi', time_viewer=True, clim=clim)

