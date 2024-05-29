#!/bin/bash

# Test expected log output from the dorado binary execution.

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [model_speed] [version] [batch size]"
    exit 1
fi

test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model_speed=${2:-"hac"}
version=${3:-"v4.2.0"}
model_complex="${model_speed}@${version}"
batch=${4:-384}
data_dir=$test_dir/data
pod5_dir=${data_dir}/pod5/dna_r10.4.1_e8.2_400bps_5khz/
output_dir_name=$(echo $RANDOM | head -c 10)
output_dir=${test_dir}/${output_dir_name}

mkdir -p $output_dir

test_output_file=$test_dir/${output_dir_name}_output.log

echo dorado basecaller test stage
$dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_dir -b ${batch} --emit-moves > $output_dir/calls.bam

samtools quickcheck -u $output_dir/calls.bam
samtools view -h $output_dir/calls.bam > $output_dir/calls.sam

# Check that the read group has the required model info in it's header
if ! grep   "^@RG" $output_dir/calls.sam | tr '\t' '\n' | grep "basecall_model" |  grep -q ${version} ; then
    echo "Output SAM file does not contain basecall model name in header!"
    grep  "^@RG" $output_dir/calls.sam
    exit 1
fi
if ! grep   "^@RG" $output_dir/calls.sam | tr '\t' '\n' | grep "modbase_models" |  grep -q "${version}_5mCG_5hmCG" ; then
    echo "Output SAM file does not contain modbase model name in header!"
    grep  "^@RG" $output_dir/calls.sam
    exit 1
fi

set +e
if $dorado_bin basecaller $model_complex $pod5_dir -b ${batch} --emit-fastq --reference $output_dir/ref.fq > $output_dir/error_condition.fq; then
    echo "Error: dorado basecaller should fail with combination of emit-fastq and reference!"
    exit 1
fi
if $dorado_bin basecaller fast@${version} $pod5_dir -b ${batch} --emit-fastq --modified-bases 5mCG_5hmCG > $output_dir/error_condition.fq; then
    echo  "Error: dorado basecaller should fail with combination of emit-fastq and modbase!"
    exit 1
fi
set -e


echo dorado summary test stage
$dorado_bin summary $output_dir/calls.bam

echo dorado basecaller mixed model complex and --modified-bases
$dorado_bin basecaller $model_complex $pod5_dir -b ${batch} --modified-bases 5mCG_5hmCG -vv > $output_dir/calls.bam
samtools view -h $output_dir/calls.bam | grep "ML:B:C,"
samtools view -h $output_dir/calls.bam | grep "MM:Z:C+h"
samtools view -h $output_dir/calls.bam | grep "MN:i:"

echo redirecting stderr to stdout: check output is still valid
# The debug layer prints to stderr to say that it's enabled, so disable it for this test.
env -u MTL_DEBUG_LAYER $dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_dir -b ${batch} --emit-moves > $output_dir/calls.bam 2>&1
samtools quickcheck -u $output_dir/calls.bam

echo dorado aligner test stage
$dorado_bin basecaller $model_complex $pod5_dir -b ${batch} --emit-fastq > $output_dir/ref.fq
$dorado_bin aligner $output_dir/ref.fq $output_dir/calls.sam > $output_dir/calls.bam
$dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_dir -b ${batch} | $dorado_bin aligner $output_dir/ref.fq > $output_dir/calls.bam
$dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_dir -b ${batch} --reference $output_dir/ref.fq > $output_dir/calls.bam
samtools quickcheck -u $output_dir/calls.bam

if ! uname -r | grep -q tegra; then
    echo dorado duplex basespace test stage
    $dorado_bin duplex basespace $data_dir/basespace/pairs.bam --threads 1 --pairs $data_dir/basespace/pairs.txt > $output_dir/calls.bam

    echo dorado in-line duplex test stage
    $dorado_bin duplex ${model_complex} $data_dir/duplex/pod5 > $output_dir/duplex_calls.bam
    samtools quickcheck -u $output_dir/duplex_calls.bam
    num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
    if [[ $num_duplex_reads -ne "2" ]]; then
        echo "Duplex basecalling missing reads."
        exit 1
    fi

    echo dorado pairs file based duplex test stage
    $dorado_bin duplex ${model_complex} $data_dir/duplex/pod5 --pairs $data_dir/duplex/pairs.txt > $output_dir/duplex_calls.bam
    samtools quickcheck -u $output_dir/duplex_calls.bam
    num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
    if [[ $num_duplex_reads -ne "2" ]]; then
        echo "Duplex basecalling missing reads."
        exit 1
    fi
fi

set +e
if ls .temp_dorado_model-* ; then 
    echo ".temp_dorado_models not cleaned"
    exit 1
fi 
set -e


rm -rf $output_dir