#!/bin/bash

# Test expected log output from the dorado binary execution.
dorado_bin=$(readlink -f $1)
test_dir=$(dirname $0)
data_dir=$test_dir/data
output_dir=${test_dir}/test_output
mkdir -p $output_dir

test_output_file=$test_dir/output.log

# Setup
$dorado_bin download --model dna_r9.4.1_e8_hac@v3.3 > /dev/null

# Test commands
$dorado_bin basecaller dna_r9.4.1_e8_hac@v3.3 $data_dir/pod5 -b 384 > $output_dir/output.bam 2>$test_output_file
grep "Reads basecalled: 1" $test_output_file
ret_val=$?
if [ $ret_val -ne 0 ]; then
    echo "Couldn't find number of reads called in output"
    exit $ret_val
fi

grep "Basecalled @ Samples/s" $test_output_file
ret_val=$?
if [ $ret_val -ne 0 ]; then
    echo "Couldn't find basecalling speed in output"
    exit $ret_val
fi
