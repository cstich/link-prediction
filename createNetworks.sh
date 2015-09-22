#!/bin/bash
echo "Creating networks..."
PARSED_DATA=($(ls $1*/*)) # Path to the parsed data files
FILES=($(ls -d $2*)) # Output path
MINIMUM_OF_MEETINGS=$3
MINIMUM_TIME=$4

for ((i=0;i<${#PARSED_DATA[@]};++i)); do
    echo "Working on:  	${PARSED_DATA[i]} ${FILES[i]}"
    rm -r "${FILES[i]}/networks/"* # Some files might linger on otherwise
    python ./buildTrainingData.py "${PARSED_DATA[i]}" "$MINIMUM_OF_MEETINGS" "$MINIMUM_TIME" "${FILES[i]}/networks/" "${FILES[i]}/university_networks/" "${FILES[i]}/all_networks/" "${FILES[i]}/weighted_networks/"  "${FILES[i]}/weighted_university_networks/"  "${FILES[i]}/weighted_all_networks/"
done
