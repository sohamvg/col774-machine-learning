#!/bin/bash
$PYTHON=python3.6

data_dir=$1
out_dir=$2
question=$3
part=$4
if [[ ${question}_${part} == "1_a" ]]; then
  $PYTHON question1a.py $data_dir $out_dir
fi