'''
Predicting classification of new data based on previously trained svc classifier

IN:
<descFile>         = csv format: headerline; col 1 = index, col 2+ = descriptors
<ZFile>            = names, means, and stds of descriptors used by svc
<svcPkl>           = pickled previously trained svc classifier

OUT:



'''

## imports
import os
import sys

import numpy as np
import numpy.matlib
from sklearn.externals import joblib


## methods

# usage
def unique(a):
    ''' return the list with duplicate elements removed '''
    return list(set(a))


def intersect(a, b):
    ''' return the intersection of two lists '''
    return list(set(a) & set(b))


def union(a, b):
    ''' return the union of two lists '''
    return list(set(a) | set(b))


## main


ZFile = "/home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/predictionsParameters/Z_score_mean_std__intersect_noflip.csv"
svcPkl = "/home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/predictionsParameters/svc.pkl"

# loading, selecting, and Z-scoring descriptors

# - loading descriptors pertaining to new sequences


# data_desc = [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00,2.50000000e+01,1.07303889e+04,3.74525000e+03,0.00000000e+00,8.47260000e-02]


def predictSVC(data_desc):

    data_desc=np.array(data_desc)
    headers=['seqIndex','netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK', '_SolventAccessibilityD1025', 'tau2_GRAR740104', 'tau4_GRAR740104', 'QSO50_GRAR740104', 'QSO29_GRAR740104']
    headers_index = headers[0]
    headers_desc = headers[1:]
    with open(ZFile, 'r') as fin:
        line = fin.readline()
        descriptors_select = line.strip().split(',')

        line = fin.readline()
        Z_means = line.strip().split(',')
        Z_means = [float(x) for x in Z_means]

        line = fin.readline()
        Z_stds = line.strip().split(',')
        Z_stds = [float(x) for x in Z_stds]

    # - extracting from descriptors only those in ZFile and alerting user to any not present


    mask = []
    for ii in range(0, len(descriptors_select)):
        try:
            idx = headers_desc.index(descriptors_select[ii])
        except ValueError:
            print('ERROR: Descriptor; aborting.')
            sys.exit(-1)
        mask.append(idx)

    headers_desc = [headers_desc[x] for x in mask]
    data_desc = data_desc[mask]
    # print ' '
    # print data_desc

    # - applying Z-scoring imported from infileZ
    Z_means = np.array(Z_means)
    Z_stds = np.array(Z_stds)

    data_desc = data_desc - Z_means
    data_desc = np.divide(data_desc, Z_stds)

    # loading classifier
    svc = joblib.load(svcPkl)

    # performing classification prediction
    distToMargin = svc.decision_function(data_desc)
    classProb = svc.predict_proba(data_desc)

    # print "prediction",'distToMargin','P(-1)','P(+1)'
    # print ' '
    if distToMargin>=0:
        return 1,distToMargin,classProb[0,0],classProb[0,1],
    else: 
        return -1,distToMargin,classProb[0,0],classProb[0,1],
