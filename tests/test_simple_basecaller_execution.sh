#!/bin/bash

# Test expected log output from the dorado binary execution.

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [4k model] [batch size] [5k model]"
    exit 1
fi

test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model_name=${2:-dna_r10.4.1_e8.2_400bps_hac@v4.1.0}
batch=${3:-384}
model_name_5k=${4:-dna_r10.4.1_e8.2_400bps_hac@v4.2.0}
data_dir=$test_dir/data
output_dir_name=$(echo $RANDOM | head -c 10)
output_dir=${test_dir}/${output_dir_name}
mkdir -p $output_dir

test_output_file=$test_dir/${output_dir_name}_output.log

echo dorado download model
$dorado_bin download --model ${model_name} --directory ${output_dir}
model=${output_dir}/${model_name}

echo dorado basecaller test stage
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --emit-fastq > $output_dir/ref.fq
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves > $output_dir/calls.bam
if ! uname -r | grep -q tegra; then
    $dorado_bin basecaller ${model} $data_dir/pod5 -x cpu --modified-bases 5mCG_5hmCG > $output_dir/calls.bam
fi
samtools quickcheck -u $output_dir/calls.bam
samtools view $output_dir/calls.bam > $output_dir/calls.sam

echo dorado summary test stage
$dorado_bin summary $output_dir/calls.bam

echo redirecting stderr to stdout: check output is still valid
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves > $output_dir/calls.bam 2>&1
samtools quickcheck -u $output_dir/calls.bam
samtools view $output_dir/calls.bam > $output_dir/calls.sam

echo dorado aligner test stage
$dorado_bin aligner $output_dir/ref.fq $output_dir/calls.sam > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG | $dorado_bin aligner $output_dir/ref.fq > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG --reference $output_dir/ref.fq > $output_dir/calls.bam
samtools quickcheck -u $output_dir/calls.bam
samtools view -h $output_dir/calls.bam > $output_dir/calls.sam

if ! uname -r | grep -q tegra; then
    echo dorado duplex basespace test stage
    $dorado_bin duplex basespace $data_dir/basespace/pairs.bam --threads 1 --pairs $data_dir/basespace/pairs.txt > $output_dir/calls.bam

    echo dorado in-line duplex test stage
    $dorado_bin download --model ${model_name_5k} --directory ${output_dir}
    duplex_model=${output_dir}/${model_name_5k}
    $dorado_bin duplex $duplex_model $data_dir/duplex/pod5 > $output_dir/duplex_calls.bam
    samtools quickcheck -u $output_dir/duplex_calls.bam
    num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
    if [[ $num_duplex_reads -ne "2" ]]; then
        echo "Duplex basecalling missing reads."
        exit 1
    fi

    echo dorado pairs file based duplex test stage
    $dorado_bin duplex $duplex_model $data_dir/duplex/pod5 --pairs $data_dir/duplex/pairs.txt > $output_dir/duplex_calls.bam
    samtools quickcheck -u $output_dir/duplex_calls.bam
    num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
    if [[ $num_duplex_reads -ne "2" ]]; then
        echo "Duplex basecalling missing reads."
        exit 1
    fi
fi

if command -v truncate > /dev/null
then
    echo dorado basecaller resume feature
    $dorado_bin basecaller -b ${batch} ${model} $data_dir/multi_read_pod5 > $output_dir/tmp.bam
    truncate -s 20K $output_dir/tmp.bam
    $dorado_bin basecaller ${model} $data_dir/multi_read_pod5 -b ${batch} --resume-from $output_dir/tmp.bam > $output_dir/calls.bam
    samtools quickcheck -u $output_dir/calls.bam
    num_reads=$(samtools view -c $output_dir/calls.bam)
    if [[ $num_reads -ne "4" ]]; then
        echo "Resumed basecalling has incorrect number of reads."
        exit 1
    fi
fi

echo dorado demux test stage
$dorado_bin demux $data_dir/barcode_demux/double_end_variant/EXP-PBC096_BC04.fastq --kit-name EXP-PBC096 --output-dir $output_dir/demux
samtools quickcheck -u $output_dir/demux/EXP-PBC096_BC04.bam
num_demuxed_reads=$(samtools view -c $output_dir/demux/EXP-PBC096_BC04.bam)
if [[ $num_demuxed_reads -ne "3" ]]; then
    echo "3 demuxed reads expected. Found ${num_demuxed_reads}"
    exit 1
fi

echo "dorado test poly(A) tail estimation"
$dorado_bin basecaller -b ${batch} ${model} $data_dir/poly_a/r10_cdna_pod5/ --estimate-poly-a > $output_dir/polya.bam
samtools quickcheck -u $output_dir/polya.bam
num_estimated_reads=$(samtools view $output_dir/polya.bam | grep pt:i: | wc -l | awk '{print $1}')
if [[ $num_estimated_reads -ne "2" ]]; then
    echo "2 poly(A) estimated reads expected. Found ${num_estimated_reads}"
    exit 1
fi

rm -rf $output_dir
