FASTA_DIR="/zhouyuyang/project/ProteinFeatureEngineering/data/fp"
OUTPUT_DIR="/zhouyuyang/project/ProteinFeatureEngineering/data/fp_iden"

# Function to remove duplicate sequences
remove_duplicate_sequences() {
    input_file="$1"
    output_file="$2"
    awk '/^>/ {if (seqlen) print seqlen; print; getline; seqlen = ""; next} {seqlen = seqlen$0} END {print seqlen}' "$input_file" | sort -u > "$output_file"

    # awk '/^>/{print;if(seen[$0]++){next}} !seen[$0]++' "$input_file" > "$output_file"
}

# Iterate over each fasta file in the input directory
for fasta_file in "$FASTA_DIR"/*.fasta; do
    filename=$(basename "$fasta_file")
    output_file="$OUTPUT_DIR/$filename"
    remove_duplicate_sequences "$fasta_file" "$output_file"
done
