ALN_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp_aln
HMM_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp_hmm
HMMBUILD=/usr/bin/hmmbuild

for file in `ls $ALN_DIR/*.aln`
do
    $HMMBUILD $HMM_DIR/$(basename $file .aln).hmm $file
done

HMM_ALL=/zhouyuyang/project/ProteinFeatureEngineering/data/fp.hmm

for file in `ls $HMM_DIR/*.hmm`
do
    cat $file >> $HMM_ALL
done
