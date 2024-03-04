

# copy in sequences to test
 seqFile='./a_AMP.txt'
 nSeq="$(wc -l < $seqFile)"

 # generate descriptors
 python ../../descriptors/descripGen_12.py "../../descriptors/aaindex/" $seqFile "./" 1 $nSeq
 descFile='descriptors.csv'
 ZFile='../../predictionsParameters/Z_score_mean_std__intersect_noflip.csv'
 svcPkl='../../predictionsParameters/svc.pkl'
    
 python ../predictSVC.py $descFile $ZFile $svcPkl

