DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp
ALN_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp_aln
CLUSTALS=/zhouyuyang/env/clustalw-2.1-linux-x86_64-libcppstatic/clustalw2

for file in `ls $DIR/*.fasta`
do
    $CLUSTALS -INFILE=$file -OUTFILE=$ALN_DIR/$(basename $file .fasta).aln
done
```