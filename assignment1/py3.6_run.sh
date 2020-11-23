#!/bin/bash
data_dir=$1
out_dir=$2
question=$3
part=$4

if [[ ${question}_${part} == "1_a" ]]; then
  python3.6 question1a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_b" ]]; then
  python3.6 question1b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_c" ]]; then
  python3.6 question1c.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_d" ]]; then
  python3.6 question1d.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_e" ]]; then
  python3.6 question1e.py $data_dir $out_dir
fi


if [[ ${question}_${part} == "2_a" ]]; then
  python3.6 question2a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "2_b" ]]; then
  python3.6 question2b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "2_c" ]]; then
  python3.6 question2c.py $data_dir $out_dir
fi


if [[ ${question}_${part} == "3_a" ]]; then
  python3.6 question3a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "3_b" ]]; then
  python3.6 question3b.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "4_a" ]]; then
  python3.6 question4a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_b" ]]; then
  python3.6 question4b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_c" ]]; then
  python3.6 question4c.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_d" ]]; then
  python3.6 question4d.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_e" ]]; then
  python3.6 question4e.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_f" ]]; then
  python3.6 question4f.py $data_dir $out_dir
fi