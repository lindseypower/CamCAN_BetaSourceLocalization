import mne 
import numpy as np
import os
from pyls import pls_regression
import pandas as pd

#Script to read to compare stc files to participant age and determine the features of the MEG source data
#that explain variance in the age #

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
     
subjectIDs = subjectData['SubjectID'].tolist()
ages = subjectData['Age_x'].tolist()

#Remove unwanted subjects from subject and age lists 
arrSubs = np.asarray(subjectIDs)
count = 0
ex_subs = ['CC520395','CC222326','CC310414','CC320568', 'CC320636', 'CC321595', 'CC510534','CC520136','CC520745', 'CC520775', 'CC621080', 'CC720304']
for x in ex_subs:
    index = np.where(arrSubs==x)[0][0]
    arrSubs = np.delete(arrSubs, index)
    del subjectIDs[index]
    del ages[index]

# Set folders and files
channelName = 'MEG0221'
dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'+ channelName +'/'
subjectDir = '/home/timb/camcan/subjects/'
stcPrefix = 'transdef_mf2pt2_rest_raw_rest_210s_cleaned-epo_restBetaEvents_DICS_fsaverage'

# Loop over all subject folders
stcs = []
for subjectID in subjectIDs:
     
    # Set file path for stc file (without Xh.stc)
    thisStcFile = os.path.join(dataDir, subjectID, stcPrefix)

    # If file exists, add the stc data to a list
    stc = mne.read_source_estimate(thisStcFile)
    stcs.append(stc.data)

stcArray = np.asarray(stcs).reshape(len(subjectIDs),20484)
ageArray = np.asarray(ages).reshape(len(subjectIDs),1)
ageArray = ageArray.astype('float32')

plsr = pls_regression(stcArray, ageArray, n_components=5)
plsr

#save plsr files to directory
plsrDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'
np.save(plsrDir+'x_weights.npy',plsr.x_weights)
np.save(plsrDir+'x_scores.npy', plsr.x_scores)
np.save(plsrDir+'y_scores.npy',plsr.y_scores)
np.save(plsrDir+'y_loadings.npy',plsr.y_loadings)
np.save(plsrDir+'pvals.npy',plsr.permres.pvals)
