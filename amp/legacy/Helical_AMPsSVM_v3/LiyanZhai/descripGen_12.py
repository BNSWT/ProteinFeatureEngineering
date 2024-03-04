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

aaindex_path = "/home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/descriptors/aaindex"

# running through sequences, generating descriptors, and writing to descriptors.csv

# seqs="GYFGGYGGGLGGIAGGLGGG"

def descripGen(seqs):
    proseq = seqs
    if ProCheck.ProteinCheck(proseq) == 0:
        print("ERROR: protein sequence %s is invalid" % (proseq))
        sys.exit()
    descNames, descValues = descripGen_bespoke(proseq)
    return descValues





