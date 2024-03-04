 
python ../seqWindowConstructor.py motherSequences.txt ./ 20 24

# copy in sequences to test
 seqFile='./subSeqs_1.txt'
 nSeq="$(wc -l < $seqFile)"

 # generate descriptors
 python ../../descriptors/descripGen_12.py "../../descriptors/aaindex/" $seqFile "./" 1 $nSeq
 descFile='descriptors.csv'
 ZFile='../../predictionsParameters/Z_score_mean_std__intersect_noflip.csv'
 svcPkl='../../predictionsParameters/svc.pkl'
    
 python ../predictSVC.py $descFile $ZFile $svcPkl

