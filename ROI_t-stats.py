import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from numpy.random import randn
from scipy import stats as stats
import pandas as pd
import seaborn as sbn

import mne

#Set folders and files
channelName = 'MEG1311'
dataDir = '/media/NAS/lpower/BetaSourceLocalization/restData/'+ channelName +'/'
subjectsDir = '/home/timb/camcan/subjects/'
stcPrefix = 'rest_DICS_stcGAvg-lh.stc'

stcFileName = os.path.join(dataDir, stcPrefix)
StatsFileName = os.path.join(dataDir, 'ROIdata/BF_rest_stats.csv')
outFileName = os.path.join(dataDir, 'DICS_rest_tstat')

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
                        u'G_pariet_inf-Supramar-lh',u'G_pariet_inf-Supramar-rh',
			u'G_parietal_sup-lh', u'G_parietal_sup-rh']

#Read ROI time course for all conditions from numpy file 
ROIData_df = pd.read_csv(StatsFileName)

labels = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s', subjects_dir=subjectsDir)
ROIlabels = []
for ROI in ROIs:
    print(ROI)
    b = [x for x in labels if x.name == ROI]
    ROIlabels.append(b[0])

#Make an array for STC data that has all zeroes. This will get filled with ROI data one label at a time
stcData = np.zeros((20484, 1))

#Pull t-stats out of ROI stats file 
tstats = ROIData_df['t-stat'].tolist()
print(tstats)

#Loop over ROIs
for ctr in np.arange(len(ROIlabels)):
    thislabel = ROIlabels[ctr]
    if 'lh' in thislabel.name:
        theseVertices = thislabel.get_vertices_used()
    else:
        theseVertices = thislabel.get_vertices_used() + 10242
    thislabelValue = tstats[ctr]
    
    #Write the tstat value to each vertex in the region of interest
    for vertex in theseVertices:
        stcData[vertex] = thislabelValue

#Put this stc data into an fsaverage stc object 
stc = mne.read_source_estimate(stcFileName)
stc.data = stcData
stc.save(outFileName)


