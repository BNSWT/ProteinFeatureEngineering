
#!/bin/bash
set -e


while IFS=, read -r col1 col2
do
    echo "I got:$col1|$col2"

    mkdir -p ./$col1
    cd 	./$col1

    echo $col2 > sequence.txt

    python ../../seqWindowConstructor.py sequence.txt ./ 20 34

    # copy in sequences to test
    seqFile='./subSeqs_1.txt'
    nSeq="$(wc -l < $seqFile)"

    # generate descriptors
    python ../../../descriptors/descripGen_12.py "../../../descriptors/aaindex/" $seqFile "./" 1 $nSeq
    descFile='descriptors.csv'
    ZFile='../../../predictionsParameters/Z_score_mean_std__intersect_noflip.csv'
    svcPkl='../../../predictionsParameters/svc.pkl'
    
    python ../../predictSVC.py $descFile $ZFile $svcPkl

    cd ../

done < wuhan_coronaVirus.csv







