"""
Code to generate descriptors using propy libraries

REFS:
propy: 		Cao et al. Bioinformatics 29 7 960 (2013) 		-- terrible paper
profeat:	Li et al. Nucleic Acids Research 34 W32 (2006) 	-- excellent description of all the descriptors employed below

PREREQS:
- python
- installation of python propy library (https://code.google.com/p/protpy/wiki/propy)

IN:		filename 			- 2 col text file, col 1 = seq id, col 2 = peptide sequence as single letter aa code
		aaindexDirPath    	- location of dirctory aaindex containing aaindex1, aaindex2, and aaindex3

OUT:	descriptors.csv		- n col text file containing one headerline with descriptor identifiers and rows of descriptors corresponding to each sequence
		(descriptors_headers.csv & descriptors_values.csv are temporary files produced during runtime and deleted at termination of code)
"""

## imports
import sys
import time

import numpy as np
from propy import AAIndex
from propy import ProCheck
from propy.PyPro import GetProDes


## methods
def _usage():
    print "USAGE: %s  <aaindexDirPath> <filename> <startIndex> <stopIndex>" % sys.argv[0]
    print "       aaindexDirPath    = location of dirctory aaindex containing aaindex1, aaindex2, and aaindex3"
    print "       filename   		= 2 col text file, col 1 = seq id, col 2 = peptide sequence as single letter aa code"
    print "       startIndex 		= # of first row to process"
    print "       stopIndex  		= # of last row to process"


def unique(a):
    ''' return the list with duplicate elements removed '''
    return list(set(a))


def intersect(a, b):
    ''' return the intersection of two lists '''
    return list(set(a) & set(b))


def union(a, b):
    ''' return the union of two lists '''
    return list(set(a) | set(b))


def descripGen_bespoke(proseq):
    v_lambda = 7  # no. of tiers in computation of PseAAC; must be shorter than minimum peptide length
    v_nlag = 30  # maximum aa sequence separation in computation of sequence-order features
    v_weight = 0.05  # weight allocated to tiered correlation descriptors in PseAAC; 0.05 recommended in [Chou, Current Proteomics, 2009, 6, 262-274]

    descNames = []
    descValues = np.empty([0, ])

    if ProCheck.ProteinCheck(proseq) == 0:
        print("ERROR: protein sequence %s is invalid" % (proseq))
        sys.exit()

    seqLength = len(proseq)

    Des = GetProDes(proseq)

    # 1: netCharge
    chargeDict = {"A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1, "I": 0, "K": 1, "L": 0, "M": 0, "N": 0,
                  "P": 0, "Q": 0, "R": 1, "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0}
    netCharge = sum([chargeDict[x] for x in proseq])

    descNames.append('netCharge')
    descValues = np.append(descValues, netCharge)

    # 2-6: FC, LW, DP, NK, AE
    dpc = Des.GetDPComp()
    for handle in ['FC', 'LW', 'DP', 'NK', 'AE']:
        descNames.append(handle)
        descValues = np.append(descValues, dpc[handle])

    # 7: pcMK
    pp = 'M'
    qq = 'K'
    Npp = sum([1 for x in proseq if x == pp])
    Nqq = sum([1 for x in proseq if x == qq])
    if Npp == 0:
        pc_pp_qq = 0
    else:
        pc_pp_qq = float(Npp) / float(Npp + Nqq)
    descNames.append('pc' + pp + qq)
    descValues = np.append(descValues, pc_pp_qq)

    # 8: _SolventAccessibilityD1025
    ctd = Des.GetCTD()
    for handle in ['_SolventAccessibilityD1025']:
        descNames.append(handle)
        descValues = np.append(descValues, ctd[handle])

    # 9-10: tau2_GRAR740104, tau4_GRAR740104
    prop = 'GRAR740104';
    AAP_dict = AAIndex.GetAAIndex23(prop, path=aaindex_path)

    socn_p = Des.GetSOCNp(maxlag=v_nlag, distancematrix=AAP_dict)

    for handle in ['tau2', 'tau4']:
        delta = float(handle[3:])
        if (delta > (seqLength - 1)):
            value = 0
        else:
            value = socn_p[handle] / float(seqLength - delta)
        descNames.append(handle + "_" + prop)
        descValues = np.append(descValues, value)

    # 11-12: QSO50_GRAR740104, QSO29_GRAR740104
    prop = 'GRAR740104';
    AAP_dict = AAIndex.GetAAIndex23(prop, path=aaindex_path)

    qso_p = Des.GetQSOp(maxlag=v_nlag, weight=v_weight, distancematrix=AAP_dict)

    for handle in ['QSO50', 'QSO29']:
        descNames.append(handle + "_" + prop)
        descValues = np.append(descValues, qso_p[handle])

    return descNames, descValues


## globals
global silentFlag
silentFlag = 1

global v_lambda
v_lambda = 7  # no. of tiers in computation of PseAAC; must be shorter than minimum peptide length

global v_nlag
v_nlag = 30  # maximum aa sequence separation in computation of sequence-order features

global v_weight
v_weight = 0.05  # weight allocated to tiered correlation descriptors in PseAAC; 0.05 recommended in [Chou, Current Proteomics, 2009, 6, 262-274]

## main

# error checking
if len(sys.argv) != 6:
    _usage()
    sys.exit(1)

aaindex_path = sys.argv[1]
inFile = sys.argv[2]
outputFolder = str(sys.argv[3])
startIndex = int(sys.argv[4])
stopIndex = int(sys.argv[5])

# loading sequences and indices into seqs and indices lists
indices = []
seqs = []
with open(inFile, 'r') as seqsFile:
    for line in seqsFile:
        a, b = line.split()
        a = int(a)
        b = str(b)
        indices.append(a)
        seqs.append(b)

if startIndex < 1:
    print "ERROR: startIndex < 1"
    sys.exit(1)

if stopIndex > len(indices):
    print("ERROR: stopIndex > # seqs in %s" % (inFile))
    sys.exit(1)

if startIndex > stopIndex:
    print "ERROR: startIndex >= stopIndex"
    sys.exit(1)

if v_lambda >= len(min(seqs, key=len)):
    print("ERROR: v_lambda = %d >= minimum seq length in %s" % (v_lambda, inFile))
    sys.exit(1)

# if v_nlag >= len(min(seqs, key=len)):
#	print("ERROR: v_nlag = %d >= minimum seq length in %s" % (v_nlag, inFile))
#	sys.exit(1)


# start timer
t_start = time.clock()

# running through sequences, generating descriptors, and writing to descriptors.csv

fout = open(outputFolder + "descriptors.csv", "w")

print("Commencing descriptor generation for selected sequences in %s" % (inFile))
for SEQIDX in range(startIndex - 1, stopIndex):

    # - print to screen
    print("    Processing sequence %d (%d more sequences remain)" % (SEQIDX + 1, stopIndex - SEQIDX - 1))

    # - selecting and verifying sequence
    proseq = seqs[SEQIDX]
    seqIndex = indices[SEQIDX]
    if ProCheck.ProteinCheck(proseq) == 0:
        print("ERROR: protein sequence %s is invalid" % (proseq))
        sys.exit()

    # - generating descriptors
    descNames, descValues = descripGen_bespoke(proseq)

    # - writing to file
    if SEQIDX == (startIndex - 1):
        fout.write('%s' % ('seqIndex,'))
        for ii in range(0, len(descNames)):
            fout.write('%s' % (descNames[ii]))
            if ii < (len(descNames) - 1):
                fout.write(',')
            else:
                fout.write('\n')

    fout.write('%s,' % (str(seqIndex)))
    for ii in range(0, len(descValues)):
        fout.write('%s' % (str(descValues[ii])))
        if ii < (len(descNames) - 1):
            fout.write(',')
        else:
            fout.write('\n')

print("DONE!")

fout.close()

# stop timer
t_stop = time.clock()
print("Elapsed time      = %.2f s" % (t_stop - t_start))
print("Time per sequence = %.2f s" % ((t_stop - t_start) / (stopIndex - startIndex + 1)))
