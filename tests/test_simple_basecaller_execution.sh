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

echo dorado download models
$dorado_bin download --model ${model_name} --directory ${output_dir}
model=${output_dir}/${model_name}
$dorado_bin download --model ${model_name_5k} --directory ${output_dir}
model_5k=${output_dir}/${model_name_5k}

echo dorado basecaller test stage
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --emit-fastq > $output_dir/REF.fq
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
$dorado_bin aligner $output_dir/REF.fq $output_dir/calls.sam > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG | $dorado_bin aligner $output_dir/REF.fq > $output_dir/calls.bam
$dorado_bin basecaller ${model} $data_dir/pod5 -b ${batch} --modified-bases 5mCG_5hmCG --reference $output_dir/REF.fq > $output_dir/calls.bam
samtools quickcheck -u $output_dir/calls.bam
samtools view -h $output_dir/calls.bam > $output_dir/calls.sam

echo dorado aligner options test stage
dorado_aligner_options_test() (
    set +e
    set +x
    which minimap2
    minimap2 --version

    # list of options and whether they affect the output
    REF=$data_dir/aligner_test/lambda_ecoli.fasta
    RDS=$data_dir/aligner_test/dataset.fastq

    RETURN=true
    touch err
    ERROR() { echo $*; RETURN=false; cat err; }
    SKIP() { echo $*; cat err; }

    OPTIONS=(""    "-k 20" "-w 100" "-I 100K" "--secondary no" "-N 1" "-r 10,100" "-Y" "--secondary-seq" "--print-aln-seq")
    CHANGES=(false true    true     false     true             true   true        true true              false            )
    for ((i = 0; i < ${#OPTIONS[@]}; i++)); do
        opt=${OPTIONS[$i]}
        echo -n "$i: with options '$opt' ... "

        # run dorado aligner
        if ! $dorado_bin aligner $opt $REF $RDS 2>err | samtools view -h 2>>err > $output_dir/dorado-$i.sam; then
            ERROR failed running dorado aligner
            continue
        fi

        # check output integrity
        if ! samtools quickcheck -u $output_dir/dorado-$i.sam 2>err; then
            ERROR failed sam check
            continue
        fi

        # sort and cut output for comparison
        sort $output_dir/dorado-$i.sam | grep -v '^@PG' | cut -f-11> $output_dir/dorado-$i.ssam

        # compare with minimap2 output
        if minimap2 -a $opt $REF $RDS 2>/dev/null > $output_dir/minimap2-$i.sam; then
            sort $output_dir/minimap2-$i.sam | grep -v '^@PG' | cut -f-11 > $output_dir/minimap2-$i.ssam
            if ! diff $output_dir/dorado-$i.ssam $output_dir/minimap2-$i.ssam > err; then
                ERROR failed comparison with minimap2 output
                continue
            fi
        else
            SKIP error running minimap2
            continue
        fi

        # check output changed
        should_change=${CHANGES[$i]}
        if diff $output_dir/dorado-$i.ssam $output_dir/dorado-0.ssam > err; then
            does_change=false
        else
            does_change=true
        fi
        if [[ $should_change != $does_change ]]; then
            $should_change && ERROR failed to change output || ERROR failed to preserve output
            continue
        fi

        echo success
    done
    $RETURN
)
dorado_aligner_options_test

if ! uname -r | grep -q tegra; then
    echo dorado duplex basespace test stage
    $dorado_bin duplex basespace $data_dir/basespace/pairs.bam --threads 1 --pairs $data_dir/basespace/pairs.txt > $output_dir/calls.bam

    echo dorado in-line duplex test stage
    $dorado_bin duplex $model_5k $data_dir/duplex/pod5 > $output_dir/duplex_calls.bam
    samtools quickcheck -u $output_dir/duplex_calls.bam
    num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
    if [[ $num_duplex_reads -ne "2" ]]; then
        echo "Duplex basecalling missing reads."
        exit 1
    fi

    echo dorado pairs file based duplex test stage
    $dorado_bin duplex $model_5k $data_dir/duplex/pod5 --pairs $data_dir/duplex/pairs.txt > $output_dir/duplex_calls.bam
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

echo dorado basecaller barcoding read groups
$dorado_bin basecaller -b ${batch} --kit-name SQK-RBK114-96 ${model_5k} $data_dir/barcode_demux/read_group_test > $output_dir/read_group_test.bam
samtools quickcheck -u $output_dir/read_group_test.bam
mkdir $output_dir/read_group_test
samtools split -u $output_dir/read_group_test/unknown.bam -f "$output_dir/read_group_test/rg_%!.bam" $output_dir/read_group_test.bam
# There should be 4 reads with BC01, 3 with BC04, and 2 unclassified groups.
expected_read_groups_BC01=4
expected_read_groups_BC04=3
expected_read_groups_unclassified=2
for bam in $output_dir/read_group_test/rg_*.bam; do
    if [[ $bam =~ "_SQK-RBK114-96_" ]]; then
        # Arrangement is |<kit>_<barcode>|, so trim the kit from the prefix and the .bam from the suffix.
        barcode=${bam#*_SQK-RBK114-96_}
        barcode=${barcode%.bam*}
    else
        barcode="unclassified"
    fi
    # Lookup expected count, defaulting to 0 if not set.
    expected=expected_read_groups_${barcode}
    expected=${!expected:-0}
    num_read_groups=$(samtools view -c ${bam})
    if [[ $num_read_groups -ne $expected ]]; then
        echo "Barcoding read group has incorrect number of reads. '${bam}': ${num_read_groups} != ${expected}"
        exit 1
    fi
done
# There shouldn't be any unknown groups.
num_read_groups=$(samtools view -c $output_dir/read_group_test/unknown.bam)
if [[ $num_read_groups -ne "0" ]]; then
    echo "Reads with unknown read groups found."
    exit 1
fi

rm -rf $output_dir
