"""
Apply filtering and Z-scoring determined on one data set to another

(1)		Extract descriptors identified in filtering

(2)		Apply same Z-scoring to the new data as was applied in the filtering procedure 

IN:
<filename> 				- csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors"
Z_score_mean_std.csv	- names, means, and stds of descriptors extracted by filter

OUT:
<filenamePrefix>_appliedFILTERandZ.<filenameSuffix>

"""

## imports
import os, re, sys
import time
import csv
import random
import math
import numpy as np
import numpy.matlib


## methods

# usage
def _usage():
	print "USAGE: %s <filename> <ZscoreFile>" % sys.argv[0]
	print "       <filename> 	= csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors"
	print "       <ZscoreFile> 	= csv format: row 1 = descriptor names, row 2 = Z score means, row 3 = Z score stds"




## main

# reading args and error checking
if len(sys.argv) != 3:
	_usage()
	sys.exit(-1)

infile = str(sys.argv[1])
infileZ = str(sys.argv[2])


# loading data
data_all=[]
with open(infile,'r') as fin:
	line = fin.readline()
	headers_all = line.strip().split(',')
	for line in fin:
		data_all.append(line.strip().split(','))

headers_file = headers_all[0]
data_file = [item[0] for item in data_all]

headers_index = headers_all[1]
data_index = [item[1] for item in data_all]

headers_resp = headers_all[2]
data_resp = [item[2] for item in data_all]
data_resp = [int(x) for x in data_resp]
data_resp = np.array(data_resp)

headers_desc = headers_all[3:]
data_desc = [item[3:] for item in data_all]
data_desc = [[float(y) for y in x] for x in data_desc]
data_desc = np.array(data_desc)



# loading descriptor names, Z score means, and Z score stds
with open(infileZ,'r') as fin:
	line = fin.readline()
	descriptors = line.strip().split(',')
	
	line = fin.readline()
	Z_means = line.strip().split(',')
	Z_means = [float(x) for x in Z_means]
	
	line = fin.readline()
	Z_stds = line.strip().split(',')
	Z_stds = [float(x) for x in Z_stds]



# extracting from descriptors in infile only those in infileZ and alerting user to any not present
mask = []
for ii in range(0,len(descriptors)):
	try:
		idx = headers_desc.index(descriptors[ii])
	except ValueError:
		print("ERROR: Descriptor %s specified in %s was not found among descriptors in %s; aborting." % (descriptors[ii], infileZ, infile))
		sys.exit(-1)
	mask.append(idx)
	
headers_desc = [headers_desc[x] for x in mask]
data_desc = data_desc[:,mask]



# applying Z-scoring imported from infileZ
Z_means = np.array(Z_means)
Z_stds = np.array(Z_stds)

data_desc = data_desc - np.matlib.repmat(Z_means, data_desc.shape[0], 1)
data_desc = np.divide(data_desc, np.matlib.repmat(Z_stds, data_desc.shape[0], 1))




# writing
outfile = infile[0:-4] + "_appliedFILTERandZ.csv"

with open(outfile,'w') as fout:
	
	fout.write(headers_file + ",")
	fout.write(headers_index + ",")
	fout.write(headers_resp + ",")
	for header in headers_desc:
		fout.write(header + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	
	for ii in range(0,data_desc.shape[0]):
		fout.write(data_file[ii] + ",")
		fout.write(data_index[ii] + ",")
		fout.write(str(data_resp[ii]) + ",")
		for jj in range(0,data_desc.shape[1]):
			fout.write(str(data_desc[ii][jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

