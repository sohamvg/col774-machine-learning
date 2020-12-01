#!/bin/sh

question=$1
train_data=$2
test_data=$3
output_file=$4

if [[ ${question} == "1" ]]; then
    python q1.py $train_data $test_data $output_file
fi

if [[ ${question} == "2" ]]; then
    python q2.py $train_data $test_data $output_file
fi

if [[ ${question} == "3" ]]; then
    python q3.py $train_data $test_data $output_file
fi