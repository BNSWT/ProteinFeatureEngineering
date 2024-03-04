First, open the terminal anywhere
Then, input the following codes and enter

```bash
conda activate AMPSVM
cd /home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/LiyanZhai
python main.py
```

results are stored in `result.txt`

> Note:
> the prediction procedure cannot stop automatically. It will exhaustively exhaust all possible sequences and make predictions. The prediction results are saved in the `result.txt` file. You can check it at any time. If you have obtained enough results, you can terminate the program with `ctrl`+`c`.

***Explanation of the result file***

- Column 1: Meaningless
- Column 2: The screened peptide sequence
- Column 3: The position of the phosphorylated amino acid
- Column 4: The phosphorylated amino acid
- Column 5: The possibility of phosphorylation

`kinase_seq.txt` is the screened sequence file.
