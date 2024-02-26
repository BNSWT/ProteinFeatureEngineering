from Bio import SeqIO
import os

def remove_duplicates(input_dir, output_dir):
    # 创建保存唯一序列的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".fasta"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            
            # 用于存储已经见过的序列
            sequences_seen = set()
            
            # 用于存储唯一序列
            unique_sequences = []
            
            # 打开输入文件，去重并保存唯一序列
            with open(input_file, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequence = str(record.seq)
                    if sequence not in sequences_seen:
                        sequences_seen.add(sequence)
                        unique_sequences.append(record)
            
            # 将唯一序列写入输出文件
            with open(output_file, "w") as output_handle:
                SeqIO.write(unique_sequences, output_handle, "fasta")

# 输入和输出目录
fasta_dir = "/zhouyuyang/project/ProteinFeatureEngineering/data/fp"
fasta_unique_dir = "/zhouyuyang/project/ProteinFeatureEngineering/data/fp_iden"

# 去重并保存唯一序列
remove_duplicates(fasta_dir, fasta_unique_dir)