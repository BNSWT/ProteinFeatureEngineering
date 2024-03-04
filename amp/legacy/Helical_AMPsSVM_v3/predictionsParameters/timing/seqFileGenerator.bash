#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Illegal number of parameters"
	exit
fi

awk '{printf("%6d %s\n",NR,$1)}' $1 > $2
