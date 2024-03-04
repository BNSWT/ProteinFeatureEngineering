"""
Randomly split data into training and test sets

python subsetSplit.py <seed> <XX> <filename1> <response1> <filename2> <response2> ...

<seed> 		= rng seed (<=0 => use system time)
<XX> 		= percentage of data to place in training set => (100-X)% in test set
<filenamei> = filename containing descriptors: 1 headerline, col 1 = id, col 2+ = descriptors
<responsei> = response value assigned to all objects in filenamei

OUT:

data_ALL.csv    = headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors
data_TRAIN.csv 	= headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors
data_TEST.csv   = headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors

"""

## imports
import os, re, sys
import time
import csv
import random
import math


## methods
def _usage():
	print "USAGE: %s <seed> <XX> <filename1> <response1> <filename2> <response2> ..." % sys.argv[0]
	print "       <seed>      = rng seed (<=0 => use system time)"
	print "       <XX> 		  = percentage of data to place in training set => (100-X)% in test set"
	print "       <filenamei> = filename containing descriptors: 1 headerline, col 1 = id, col 2+ = descriptors"
	print "       <responsei> = response value assigned to all objects in <filenamei>"
	print "-> Need at least two files to be meaningful or otherwise all response variables are identical"


## main

	
# error checking
if len(sys.argv) < 7:
	_usage()
	sys.exit(-1)

seed = int(sys.argv[1])

traningFraction = float(sys.argv[2])/100
testFraction = 1.0 - traningFraction

filenames = []
responses = []
for ii in range(3,len(sys.argv),2):
	filenames.append(sys.argv[ii])
	responses.append(sys.argv[ii+1])


# checking descriptor headers consistent between all files
for ii in range(0,len(filenames)):
	with open(filenames[ii],'rt') as f:
		if ii==0:
			headers = f.readline().strip().split(',')
		else:
			headersTmp = f.readline().strip().split(',')
			if headersTmp != headers:
				print("ERROR: Headers in %s are not identical to headers in %s; aborting." % (filenames[0], filenames[ii]))
				sys.exit(-1)

headers = headers[1:]
headers.insert(0, "response")
headers.insert(0, "seqIndex")
headers.insert(0, "descFile")


# loading descriptors and composing into single array
data_col1 = []
data_col2 = []
data_col3 = []
data_col4plus = []
for ii in range(0,len(filenames)):
	with open(filenames[ii],'r') as fin:
		line = fin.readline()	# rejecting headerline
		for line in fin:
			line = line.strip()
			data_col4plus.append(line.split(','))
			data_col3.append(responses[ii])
			data_col2.append(data_col4plus[-1][0])
			data_col1.append(filenames[ii])
			data_col4plus[-1] = data_col4plus[-1][1:]	# eliminating index column since this is now in data_col2


# writing composed array to file			
with open("data_ALL.csv",'w') as fout:
	for header in headers:
		fout.write(header + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	for ii in range(0,len(data_col1)):
		fout.write(data_col1[ii] + ",")
		fout.write(data_col2[ii] + ",")
		fout.write(data_col3[ii] + ",")
		for jj in range(0,len(data_col4plus[ii])):
			fout.write(data_col4plus[ii][jj] + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")


# random partitioning into training and test subsets and writing to file

idxMax_TRAIN = int(math.floor(traningFraction*len(data_col1)))
idxMax_TEST = len(data_col1)

if seed > 0:
	random.seed(seed)
index = range(0,len(data_col1))
random.shuffle(index)

index_TRAIN = index[:idxMax_TRAIN]
index_TEST = index[idxMax_TRAIN:]

with open("data_TRAIN.csv",'w') as fout:
	for header in headers:
		fout.write(header + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	for ii in index_TRAIN:
		fout.write(data_col1[ii] + ",")
		fout.write(data_col2[ii] + ",")
		fout.write(data_col3[ii] + ",")
		for jj in range(0,len(data_col4plus[ii])):
			fout.write(data_col4plus[ii][jj] + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

with open("data_TEST.csv",'w') as fout:
	for header in headers:
		fout.write(header + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	for ii in index_TEST:
		fout.write(data_col1[ii] + ",")
		fout.write(data_col2[ii] + ",")
		fout.write(data_col3[ii] + ",")
		for jj in range(0,len(data_col4plus[ii])):
			fout.write(data_col4plus[ii][jj] + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")
