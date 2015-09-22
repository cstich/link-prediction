#!/bin/sh
FILES=$(ls -d $1*) # Path to the training and testing files
OUTPUT=$3

rm "$OUTPUT/modelTimeseriesData.ssv"
rm "$OUTPUT/featureImportance.ssv"

for f in $FILES
    do
	python ./predictions.py $f "data/met_1/30_0/weighted_networks/" $2 $3 $4
    done
