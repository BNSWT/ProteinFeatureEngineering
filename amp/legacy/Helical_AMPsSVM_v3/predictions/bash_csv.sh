
#!/bin/bash
set -e

c=1
echo $c


while IFS=, read -r col1 col2
do
    echo "I got:$col1|$col2"

    mkdir -p ./$col1
    cd 	./$col1
    c=c+1

    #echo "HOLA" > sequence.txt


    #python seqWindowConstructor.py motherSequences.txt 10 25
    #cd ../

done < HydraGenes_Curated.csv

echo $c



