#!/bin/bash

set -ex

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dorado executable> <pod5_dir> [device_string]"
    echo "  <pod5_dir> should contain large input files"
    exit 1
fi

pod5_dir=${2}
device_string=${3:-"auto"}
echo "Using device string -x $device_string"
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)

echo Running benchmarks for models of interest
for model_name in \
        dna_r10.4.1_e8.2_400bps_fast@v5.2.0 \
        dna_r10.4.1_e8.2_400bps_hac@v5.2.0 \
        dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
        dna_r10.4.1_e8.2_400bps_hac@v6.1.0 \
        dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
        rna004_fast@v6.0.0 \
        rna004_hac@v6.0.0 \
        rna004_sup@v6.0.0 \
        ; do
    echo $model_name;
    $dorado_bin download --model $model_name
    $dorado_bin basecaller \
        $model_name $pod5_dir \
        -x $device_string \
        --skip-model-compatibility-check \
        --batchsize-benchmarks-file "$(hostname).csv" \
        --run-batchsize-benchmarks break \
        > /dev/null
done
