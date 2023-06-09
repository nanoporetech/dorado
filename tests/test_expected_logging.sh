#!/bin/bash

# Test expected log output from the dorado binary execution.

set -ex
set -o pipefail

test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model=${2:-dna_r9.4.1_e8_hac@v3.3}
batch=${3:-384}
data_dir=$test_dir/data
output_dir=${test_dir}/test_output
mkdir -p $output_dir

test_output_file=$test_dir/output.log

# Setup
$dorado_bin download --model ${model} > /dev/null

# Test commands
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} > $output_dir/output.bam 2>$test_output_file
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
