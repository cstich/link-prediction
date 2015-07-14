#!/bin/sh
FILES=$(ls -d $1*) # Path to the training and testing files

for f in $FILES
    do
	python ./predictions.py $f
    done
