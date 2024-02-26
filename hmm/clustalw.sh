FASTA_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp_iden
ALN_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fp_aln
CLUSTALS=/zhouyuyang/env/clustalw-2.1-linux-x86_64-libcppstatic/clustalw2

find $FASTA_DIR -name "*.fasta" | xargs -I {} basename {} .fasta | xargs -I {} $CLUSTALS -INFILE="$FASTA_DIR/{}.fasta" -OUTFILE="$ALN_DIR/{}.aln"