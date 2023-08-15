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

echo dorado basecaller test stage
$dorado_bin download --model ${model}
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --emit-fastq > $output_dir/ref.fq
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG --emit-moves > $output_dir/calls.bam
if ! uname -r | grep -q tegra; then
    $dorado_bin basecaller ${model} $data_dir/pod5 -x cpu --modified-bases 5mCG > $output_dir/calls.bam
fi
samtools quickcheck -u $output_dir/calls.bam
samtools view $output_dir/calls.bam > $output_dir/calls.sam

echo dorado summary test stage
$dorado_bin summary $output_dir/calls.bam

echo redirecting stderr to stdout: check output is still valid
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG --emit-moves > $output_dir/calls.bam 2>&1
samtools quickcheck -u $output_dir/calls.bam
samtools view $output_dir/calls.bam > $output_dir/calls.sam

echo dorado aligner test stage
$dorado_bin aligner $output_dir/ref.fq $output_dir/calls.sam > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG | $dorado_bin aligner $output_dir/ref.fq > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG --reference $output_dir/ref.fq > $output_dir/calls.bam
samtools quickcheck -u $output_dir/calls.bam
samtools view -h $output_dir/calls.bam > $output_dir/calls.sam

echo dorado duplex basespace test stage
$dorado_bin duplex basespace $data_dir/basespace/pairs.bam --threads 1 --pairs $data_dir/basespace/pairs.txt > $output_dir/calls.bam

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

rm -rf $output_dir
