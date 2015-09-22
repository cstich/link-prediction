#!/bin/bash
echo "Creating features and candidates..."
PARSED_DATA=($(ls -d $1*)) # Path to the parsed data files
FILES=($(ls -d $2*)) # Path to the features and networks files
LENGTH_OF_TEST_USERS=$3

for ((i=0;i<${#PARSED_DATA[@]};++i)); do
    rm -r "${FILES[i]}/features/"* # Some files might linger on otherwise
    echo "Working on:  	${PARSED_DATA[i]} ${FILES[i]}"
    python ./selectCandidatesAndFeatures.py "${PARSED_DATA[i]}" "${FILES[i]}/networks/" "0" "$LENGTH_OF_TEST_USERS" "${FILES[i]}/features/"
done
