import pandas as pd
import numpy as np
import random
import sys 
sys.path.append("/home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/LiyanZhai")
from descripGen_12 import descripGen
from predictSVC import predictSVC
from predictPTM import predPTM
import time
# input
target_seq="GIIGGIGGGIGGIIGGIGGG"
change_seqpos=[2,3,6,10,13,14,17] # from 1 
fix_seqpos=[]

resi_positive=["R","H","K"]
resi_negative=["D","E"]
resi_polar=["S","T","N","Q"]
resi_hydrophobic=["A","V","I","L","M","F","Y","W"]

resi_nohydrophobic=resi_positive+resi_negative+resi_polar

# input preprocess
target_seq=np.array(list(target_seq))
change_seqpos=np.array(change_seqpos)-1
change_seqpos.astype(int)
fix_seqpos=np.array(fix_seqpos)-1
change_seqpos=np.setdiff1d(change_seqpos,fix_seqpos)

# R-X-X-S-X-X-R

def generate_random_array(filter, change_seqpos):
    ''' 
    Randomly generate an array of specified length
    filter: Alternative mutation sequences
    change_seqpos: seqpos needed to change
    '''
    random_array = [random.choice(filter) for _ in range(len(change_seqpos))]
    return random_array

def output_seq(seq,output=False):
    """ 
    output the mutated sequence 
    seq: sequence needed to be output
    output: if or not output to a txt file
    """
    process_seq=[]
    for i in range(len(seq)):
        process_seq.append(''.join(seq[i]))
    
    if output==True:
        np.savetxt("result.txt",process_seq,"%s")

    return process_seq

def genseq(filter,change_seqpos,target_seq):
    ''' 
    return: generated sequence
    '''
    tmpseqs=generate_random_array(filter=filter,change_seqpos=change_seqpos)
    target_seq[change_seqpos]=tmpseqs
    return ''.join(target_seq)

def kinase_filter(seq):
    """ 
    Determine whether it contains a characteristic sequence
    "R-X-X-S-X-X-R"
    """
    for i in range(len(seq)-7):
        if seq[i]=='R' and seq[i+3]== 'S' and seq[i+6]=='R':
            return True,i
    
    return False,0

""" 
phosphorylation sites filter
"""
n=100000000
phos_score=pd.DataFrame()
kinase_tmp=[]
start_time=time.time()


for i in range(n):

    if i%10000==0:
        end_time=time.time()
        remain=(end_time-start_time)*((n-i)/10000)
        print "{} iterations had been done, Remaining time is {:.2f} s".format(i,remain)
        start_time=time.time()
        phos_score.to_csv("result.txt")

    nohydrophobic_changepos=np.setdiff1d(np.arange(len(target_seq)),change_seqpos)
    gseqs=np.array(list(genseq(resi_hydrophobic,change_seqpos,target_seq)))
    gseqs=genseq(resi_nohydrophobic,nohydrophobic_changepos,gseqs)
    
    pred,idx=kinase_filter(gseqs)

    if pred:
        kinase_tmp.append(gseqs)
        predAMP=predictSVC(descripGen(gseqs))[0]

        if predAMP==1:
            phos_score_tmp=predPTM(gseqs,score=0.85)
            if phos_score_tmp.empty==True:
                pass
            else:
                if (phos_score_tmp[1].values-1).any() in [idx,idx+3,idx+6]:

                    phos_score=phos_score.append(phos_score_tmp)


phos_score.to_csv("result.txt")

with open("kinase_seq.txt","w+") as f:
    for i in kinase_tmp:
        f.write(i+"\n")
f.close()

print phos_score

""" 
AMP classifier
"""
# n=10000
# data=[]
# for i in range(n):
#     gseqs=genseq(resi_hydrophobic,change_seqpos,target_seq)
#     test_descripGen=descripGen(gseqs)
#     pred=predictSVC(test_descripGen)[0]
#     if pred==1:
#         data.append([gseqs])
    
# print(data)
# print(len(data))

