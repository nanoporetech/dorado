#!/bin/bash

# Test expected log output from the dorado binary execution.

set -ex
set -o pipefail

test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model=${2:-dna_r9.4.1_e8_hac@v3.3}
batch=${3:-384}
data_dir=$test_dir/data
output_dir_name=$(echo $RANDOM | head -c 10)
output_dir=${test_dir}/${output_dir_name}
mkdir -p $output_dir

test_output_file=$test_dir/output.log

# Setup
$dorado_bin download --model ${model} --directory . > /dev/null
ret_val=$?
if [ $ret_val -ne 0 ]; then
    echo "Error: Failed to download model"
    exit $ret_val
fi

if [[ ! -e "${model}" ]]; then 
    echo "Error: Model does not exist"
    exit 1
fi

# Test commands
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} > $output_dir/output.bam 2>$test_output_file
grep "Simplex reads basecalled: 1" $test_output_file
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

$dorado_bin demux $data_dir/barcode_demux/double_end_variant/EXP-PBC096_BC04.fastq --kit-name EXP-PBC096 --output-dir $output_dir/demux 2>$test_output_file
grep "reads demuxed @ classifications/s" $test_output_file
ret_val=$?
if [ $ret_val -ne 0 ]; then
    echo "Couldn't find barcode demuxing speed in output"
    exit $ret_val
fi
