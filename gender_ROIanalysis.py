import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from numpy.random import randn
from scipy import stats as stats
import pandas as pd
import seaborn as sbn

import mne

###############################
# Find subjects to be analysed
homeDir = '/media/NAS/lpower/camcan/'
dataDir = homeDir + 'spectralEvents/task/MEG0221'
camcanCSV = dataDir + '/spectralEventAnalysis.csv'
subjectData = pd.read_csv(camcanCSV)
# Take only subjects with more than 55 epochs
subjectData = subjectData[subjectData['numEpochs'] > 55]
# Drop subjects with MR files missing
subjectData = subjectData.drop(subjectData[subjectData['bemExists'] == False].index)
subjectData = subjectData.drop(subjectData[subjectData['srcExists'] == False].index)
subjectData = subjectData.drop(subjectData[subjectData['transExists'] == False].index)
     
gender1 = subjectData[subjectData['Gender_x'] == 1]
gender2 = subjectData[subjectData['Gender_x'] == 2]
subs1 = gender1['SubjectID'].tolist()
subs2 = gender2['SubjectID'].tolist()
ages1 = gender1['Age_x'].tolist()
ages2 = gender2['Age_x'].tolist()
     
#Remove unwanted subjects from subject and age lists 

arrSubs = np.asarray(subs1)
count = 0
ex_subs = ['CC520395','CC222326','CC310414','CC320568', 'CC320636', 'CC321595', 'CC510534','CC520136','CC520745', 'CC520775', 'CC621080', 'CC720304']
for x in ex_subs:
    index = np.where(arrSubs==x)[0]
    if len(index) > 0: 
        index = index[0]
        arrSubs = np.delete(arrSubs, index)
        del subs1[index]
        del ages1[index]

mriID = 'fsaverage'


#Set folders and files
channelName = 'MEG0221'
dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'+ channelName +'/'
subjectsDir = '/home/timb/camcan/subjects/'
stcPrefix = 'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS_fsaverage'

ROITimeCourseDataFrameFileName1 = os.path.join(dataDir, 'ROIdata', 'DICS_ROITimeCourses_male.csv')
timeFileName = os.path.join(dataDir, 'ROIdata', 'times.npy')

plotOK = True

#################################
# Analysis starts here

# Note: see Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature,
#	
#	Labels of interest are (per hemisphere, indexed from 1, aparc.a2009s)
#		28: Postcentral gyrus
#       29: Precentral gyrus
#       15: Middle frontal gyrus
#       16: Superior frontal gyrus
#       26: Supramarginal gyrus
#       27: Superior parietal lobule
#
# labels list is not in the same indexed order, must find these regions by name
ROIs = [u'G_postcentral-lh', u'G_postcentral-rh',
			u'G_precentral-lh', u'G_precentral-rh',
			u'G_front_middle-lh', u'G_front_middle-rh',
			u'G_front_sup-lh', u'G_front_sup-rh',
			u'G_pariet_inf-Supramar-lh', u'G_pariet_inf-Supramar-rh',
			u'G_parietal_sup-lh', u'G_parietal_sup-rh']

# Read all stc files
allData = []
dfs = []

#########
#SUBSET HACK
#subjects = [subjects[0]]
#mriIDs = [mriIDs[0]]
#conds = [conds[0]]
#########
counter = 0
for subjectID in subs1:
    
    #Read in source space for this subject
    #srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    #src = mne.read_source_spaces(srcFif)
    
    src = mne.setup_source_space(mriID, subjects_dir=subjectsDir,spacing='all',add_dist=False, n_jobs=3)

    ## Pull the labels for all anatomically annotated regions
    labels = mne.read_labels_from_annot(mriID, parc='aparc.a2009s', subjects_dir=subjectsDir)

    ##### Pull the labels for the ROIs only!!######
    subLabels = []
    for ROI in ROIs:
        b = [x for x in labels if x.name == ROI]
        if len(b) > 0:
            subLabels.append(b[0])

    # Read the STC file and morph to fsaverage
    thisStcFile = os.path.join(dataDir, subjectID, stcPrefix)
    stc = mne.read_source_estimate(thisStcFile)
		
    # Extract source estimate time course for each ROI (16 x 126) 
    #		Positive definite data so just take the mean
    label_ts = mne.extract_label_time_course(stc, labels, src, mode='mean',return_generator=True)#Might have to take all of the labels in order for this to work 

    # Write data to a dataframe (for seaborn plotting and stats)
    for i in np.arange(len(ROIs)):
        df = pd.DataFrame.from_dict({'Age': ages1[counter], 'Current': label_ts[i,:]})
        df["SubjectID"] = subjectID
        df["ROI"] = ROIs[i]
        dfs.append(df)
		
    allData.append(label_ts)
    counter = counter+1

one_df = pd.concat(dfs)

# Convert all ROI data to a numpy array (subjects x vertices x time)
one_df.to_csv(ROITimeCourseDataFrameFileName1)

'''
# Make some plots
if plotOK:
	# Plot differences between absolute dSPM for important contrasts per ROI
	#	lineplot of means with confidence interval shaded
	diffDf = pd.concat([diff1Df, diff2Df])
	for hemi in np.arange(2):
		# Grab ROIs for this hemisphere
		hemiROIs = ROIs[hemi::2]  
		# Make a plot with a subplot per ROI
		fig, ax = plt.subplots(2,4)
		ax = ax.reshape(-1)  
		# Make a plot across subjects, between conditions for each ROI
		for ctr in np.arange(8):
			print('Plotting data for ROI: ' + str(ctr))
			thisROI = hemiROIs[ctr]
			dfSub = diffDf[diffDf["ROI"] == thisROI]
			sbn.lineplot(x="Time", y="Current", hue="Condition", style="Condition", 
				data=dfSub, ax=ax[ctr], palette="colorblind")#, 
				#estimator=None, units="SubjectID")
			ax[ctr].set_title(thisROI)
			ax[ctr].set_ylim((-2,2))
			ax[ctr].set_xlim((0,0.3))
			ax[ctr].grid(True)
		ax = ax.reshape(4,2)
		plt.show()
    
	# Plot absolute dSPM for Mono conditon per ROI (to see when/where main activation occurs)
	#	lineplot of means with confidence interval shaded
	for hemi in np.arange(2):
		# Grab ROIs for this hemisphere
		hemiROIs = ROIs[hemi::2]  
		# Make a plot with a subplot per ROI
		fig, ax = plt.subplots(2,4)
		ax = ax.reshape(-1)  
		# Make a plot across subjects, between conditions for each ROI
		for ctr in np.arange(8):
			print('Plotting data for ROI: ' + str(ctr))
			thisROI = hemiROIs[ctr]
			dfSub = allDf[allDf["ROI"] == thisROI]
			dfSub = dfSub[dfSub["Condition"] == "Mono"]
			sbn.lineplot(x="Time", y="Current",  
				data=dfSub, ax=ax[ctr], palette="colorblind")#, 
				#estimator=None, units="SubjectID")
			ax[ctr].set_title(thisROI)
			ax[ctr].set_ylim((0,6))
			ax[ctr].set_xlim((0,0.3))
			ax[ctr].grid(True)
		ax = ax.reshape(4,2)
		plt.show()
    '''
	
