#!/bin/bash
set -e

source activate AMPenv_SVM

# -----
# 1_descriptors
# -----

cd ./descriptors/

# make sure text files are saved as tab-delimited text if exporting from Excel
inputFile_AMP="seqs_AMP.txt"
inputFile_decoy="seqs_decoy.txt"
numSeq_AMP="$(wc -l < $inputFile_AMP)"
numSeq_decoy="$(wc -l < $inputFile_decoy)"


python descripGen_14.py "./aaindex/" "./seqs_AMP.txt" "../" 1 $numSeq_AMP
mv ../descriptors.csv ../descriptors_AMP.csv

python descripGen_14.py "./aaindex/" "./seqs_decoy.txt" "../" 1 $numSeq_decoy
mv ../descriptors.csv ../descriptors_decoy.csv


# -----
# 2_split
# -----

cd ..


seed=200184
percentTrain=85

python subsetSplit.py $seed $percentTrain "descriptors_AMP_2.csv" 1 "descriptors_decoy_2.csv" -1

# -----
# 4_filter_L1SVC
# -----

cd ../filter_L1SVC/

cp ../data_TRAIN.csv ./

infile="data_TRAIN.csv"
seed=200184
alpha_MWU=1.0
rho_crit=0.95
svc_tol=1E-4
svc_maxIter=1E6
n_CV=15
CVTestFraction=0.2
Cmin=1E-3
Cmax=1E1
nC=15
optFtol=1E-2
nDummy=10

python filter_L1SVC.py $infile $seed $alpha_MWU $rho_crit $svc_tol $svc_maxIter $n_CV $CVTestFraction $Cmin $Cmax $nC $optFtol $nDummy


# -----
# 5_filterApply
# -----

cd filterOnly
cp ../data_TEST.csv ./
cp ../filter_L1SVC/Z_score_mean_std__filterOnly.csv ./
python applyFilter.py "data_TEST.csv" "Z_score_mean_std__filterOnly.csv"
cd ..

# -----
# 6a_linearSVC
# -----


seed=200184
svc_tol=1E-4
svc_maxIter=1E7
n_CV=15
CVTestFraction=0.2
Cmin=1E-4
Cmax=1E3
nC=20
optFtol=1E-2

trainingData="./filter_L1SVC/data_TRAIN_FILTERandZ__filterOnly.csv"
testData="./filterOnly/data_TEST_appliedFILTERandZ.csv"
python linearSVC.py $trainingData $testData $seed $svc_tol $svc_maxIter $n_CV $CVTestFraction $Cmin $Cmax $nC $optFtol


# -----

source deactivate


# ----
# Predictions
# ----


cd ..

# copy in sequences to test
seqFile='./subSeqs_1.txt'
nSeq="$(wc -l < $seqFile)"

# generate descriptors
python ../../descriptors/descripGen_14.py "../../descriptors/aaindex/" $seqFile "./" 1 $nSeq
descFile='descriptors_2.csv'

# copy in Z-scoring file used to select and Z-score descriptors
cp ../../filter_L1SVC/Z_score_mean_std__filterOnly.csv ./
ZFile='Z_score_mean_std__filterOnly.csv'

# copy in pickled classifier of choice
cp ../../svc.pkl* ./
svcPkl='svc.pkl'

python ../predictSVC.py $descFile $ZFile $svcPkl
