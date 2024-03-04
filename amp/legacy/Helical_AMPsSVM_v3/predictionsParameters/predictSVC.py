'''
Predicting classification of new data based on previously trained svc classifier

IN:
<descFile>         = csv format: headerline; col 1 = index, col 2+ = descriptors
<ZFile>            = names, means, and stds of descriptors used by svc
<svcPkl>           = pickled previously trained svc classifier

OUT:



'''

## imports
import os, re, sys, time
import random, math

import numpy as np
import numpy.matlib

import scipy.optimize
import scipy.stats

import pickle as pkl

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics


## methods

# usage
def _usage():
	print 'USAGE: %s <descFile> <ZFile> <svcPkl>' % sys.argv[0]
	print '       <descFile>         = csv format: headerline; col 1 = index, col 2+ = descriptors'
	print '       <ZFile>            = names, means, and stds of descriptors used by svc'
	print '       <svcPkl>           = pickled previously trained svc classifier'

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

# reading args and error checking
if len(sys.argv) != 4:
	_usage()
	sys.exit(-1)

descFile = str(sys.argv[1])
ZFile = str(sys.argv[2])
svcPkl = str(sys.argv[3])




# loading, selecting, and Z-scoring descriptors

# - loading descriptors pertaining to new sequences
print('')
print('Loading data from %s...' % (descFile))

data=[]
with open(descFile,'r') as fin:
	line = fin.readline()
	headers = line.strip().split(',')
	for line in fin:
		data.append(line.strip().split(','))

headers_index = headers[0]
data_index = [item[0] for item in data]

headers_desc = headers[1:]
data_desc = [item[1:] for item in data]
data_desc = [[float(y) for y in x] for x in data_desc]
data_desc = np.array(data_desc)

print('Loaded')
print('')


# - loading descriptor names, Z score means, and Z score stds
print('Loading descriptor list, means, and stds from %s...' % (ZFile))

with open(ZFile,'r') as fin:
	line = fin.readline()
	descriptors_select = line.strip().split(',')

	line = fin.readline()
	Z_means = line.strip().split(',')
	Z_means = [float(x) for x in Z_means]

	line = fin.readline()
	Z_stds = line.strip().split(',')
	Z_stds = [float(x) for x in Z_stds]

print('Loaded')
print('')


# - extracting from descriptors only those in ZFile and alerting user to any not present
print('Selecting and Z-scoring descriptors...')

mask = []
for ii in range(0,len(descriptors_select)):
	try:
		idx = headers_desc.index(descriptors_select[ii])
	except ValueError:
		print('ERROR: Descriptor %s specified in %s was not found among descriptors in %s; aborting.' % (descriptors_select[ii], ZFile, descFile))
		sys.exit(-1)
	mask.append(idx)

headers_desc = [headers_desc[x] for x in mask]
data_desc = data_desc[:,mask]


# - applying Z-scoring imported from infileZ
Z_means = np.array(Z_means)
Z_stds = np.array(Z_stds)

data_desc = data_desc - np.matlib.repmat(Z_means, data_desc.shape[0], 1)
data_desc = np.divide(data_desc, np.matlib.repmat(Z_stds, data_desc.shape[0], 1))

print('Loaded')
print('')



# loading classifier
print('Loading classifier from %s...' % (svcPkl))

svc = joblib.load(svcPkl)

print('Loaded')
print('')



# performing classification prediction
print('Performing classification predictions and writing to %s...' % (descFile[0:-4] + '_PREDICTIONS.csv'))

distToMargin = svc.decision_function(data_desc)
classProb = svc.predict_proba(data_desc)

idx_sort = np.argsort(distToMargin)
idx_sort = idx_sort[::-1]

with open(descFile[0:-4] + '_PREDICTIONS.csv','w') as fout:

	fout.write(headers_index + ',')
	fout.write("prediction" + ",")
	fout.write('distToMargin' + ',')
	fout.write('P(-1)' + ',')
	fout.write('P(+1)' + ',')
	fout.seek(-1, os.SEEK_END)
	fout.truncate()
	fout.write('\n')

	for ii in idx_sort:
		fout.write(data_index[ii] + ',')
		if distToMargin[ii] >= 0:
			fout.write(str(1) + ",")
		else:
			fout.write(str(-1) + ",")
		fout.write(str(distToMargin[ii]) + ',')
		fout.write(str(classProb[ii,0]) + ',')
		fout.write(str(classProb[ii,1]) + ',')
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write('\n')

with open(descFile[0:-4] + '_PREDICTIONS_unsorted.csv','w') as fout:

	fout.write(headers_index + ',')
	fout.write("prediction" + ",")
	fout.write('distToMargin' + ',')
	fout.write('P(-1)' + ',')
	fout.write('P(+1)' + ',')
	fout.seek(-1, os.SEEK_END)
	fout.truncate()
	fout.write('\n')

	for ii in range(0,len(distToMargin)):
		fout.write(data_index[ii] + ',')
		if distToMargin[ii] >= 0:
			fout.write(str(1) + ",")
		else:
			fout.write(str(-1) + ",")
		fout.write(str(distToMargin[ii]) + ',')
		fout.write(str(classProb[ii,0]) + ',')
		fout.write(str(classProb[ii,1]) + ',')
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write('\n')

print('Predictions complete')
print('')




print('DONE!')
print('')
