FASTA_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/fasta
RES_DIR=/zhouyuyang/project/ProteinFeatureEngineering/data/hmm_res
HMM_ALL=/zhouyuyang/project/ProteinFeatureEngineering/data/fp.hmm
HMMSEARCH=/usr/bin/hmmsearch
    
find $FASTA_DIR -name "*.fasta" | xargs -I {} basename {} .fasta | xargs -I {} $HMMSEARCH --domE 1e-5 --domtblout $RES_DIR/{}.txt $HMM_ALL $FASTA_DIR/{}.fasta
