#!/bin/bash

# Integration tests for Dorado Correct.
# Note: To disable these tests, set the following env variable: "NO_TEST_DORADO_CORRECT=1".

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable>"
    exit 1
fi

# Do nothing if this env variable is set.
if [[ "${NO_TEST_DORADO_CORRECT}" == "1" ]]; then
    exit 0
fi

# CLI options.
test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
cli_out_dir="$2"

# Test data directory.
data_dir=$test_dir/data

# Output directory - allow override from the CLI.
if [[ ${cli_out_dir} != "" ]]; then
    output_dir=${cli_out_dir}
else
    output_dir_name=test_output_dc_$(echo $RANDOM | head -c 10)
    output_dir=${test_dir}/${output_dir_name}
fi

mkdir -p $output_dir

# Test dorado correct with auto detected platform. If that fails run on cpu.
# Create a folder for the new tests.
output_dir_correct_root=${output_dir}/correct
mkdir -p $output_dir_correct_root

### TEST: Check if Dorado Correct generates expected consensus results on small data. This test
###         compares: alignment accuracy vs small HG002 reference chunk, number of aligned reads and
###         number of bases produced by consensus.
###
### Note: This test will likely fail once the model is retrained or if there are any algorithmic/bugfix changes.
###         It is intentionally made strict to detect any changes (i.e. expecting identical accuracy), but the
###         actual test is easy to update since it compares only the high level stats.
#
output_dir_correct=${output_dir_correct_root}/test-01
mkdir -p ${output_dir_correct}
#
# Run Dorado Correct.
$dorado_bin correct $data_dir/read_correction/reads.fq -v > $output_dir_correct/corrected_reads.fasta
#
# Align the corrected reads to the reference portion.
$dorado_bin aligner --mm2-opts "-x map-ont" $data_dir/read_correction/ref.fasta $output_dir_correct/corrected_reads.fasta -o $output_dir_correct/ --emit-summary
#
# Sort the lines because the output of Correct is not stable.
sort $data_dir/read_correction/expected.alignment_summary.txt > $output_dir_correct/sorted.expected.alignment_summary.txt
sort $output_dir_correct/alignment_summary.txt > $output_dir_correct/sorted.alignment_summary.txt
#
# Test the accuracy and number of sequences in the output.
set +e
result=$(diff $output_dir_correct/sorted.expected.alignment_summary.txt $output_dir_correct/sorted.alignment_summary.txt | wc -l | awk '{ print $1 }')
set -e
if [[ $result -ne "0" ]]; then
    echo "Dorado correct alignment accuracy does not match expected results."
    diff $output_dir_correct/sorted.expected.alignment_summary.txt $output_dir_correct/sorted.alignment_summary.txt
    exit 1
fi
#
# Test that the sequences have the same length.
samtools faidx $output_dir_correct/corrected_reads.fasta
cut -f 1,2 $output_dir_correct/corrected_reads.fasta.fai | sort > $output_dir_correct/corrected_reads.fasta.seq_lengths.csv
set +e
result=$(diff $data_dir/read_correction/expected.seq_lengths.csv $output_dir_correct/corrected_reads.fasta.seq_lengths.csv | wc -l | awk '{ print $1 }')
set -e
if [[ $result -ne "0" ]]; then
    echo "Dorado correct sequence length do not match expected sequence lengths."
    diff $data_dir/read_correction/expected.seq_lengths.csv $output_dir_correct/corrected_reads.fasta.seq_lengths.csv
    exit 1
fi

# Test if nonexistent input reads file will fail gracefully. This test _should_ fail, that's why we deactivate the -e.
#
output_dir_correct=${output_dir_correct_root}/test-02
mkdir -p ${output_dir_correct}
#
set +e
$dorado_bin correct nonexistent.fastq -v > $output_dir_correct/corrected_reads.fasta 2> $output_dir_correct/corrected_reads.fasta.stderr
error_matched=$(grep "\[error\] Input reads file nonexistent.fastq does not exist!" $output_dir_correct/corrected_reads.fasta.stderr | wc -l | awk '{ print $1 }')
set -e
if [[ $error_matched -ne "1" ]]; then
    echo "Dorado correct does not fail on non-existent reads input file!"
    exit 1
fi

# Test if nonexistent user-specified model path will fail gracefully. This test _should_ fail, that's why we deactivate the -e.
#
output_dir_correct=${output_dir_correct_root}/test-03
mkdir -p ${output_dir_correct}
#
set +e
$dorado_bin correct $data_dir/read_correction/reads.fq -v --model-path nonexistent/dir/ > $output_dir_correct/corrected_reads.fasta 2> $output_dir_correct/corrected_reads.fasta.stderr
error_matched=$(grep "\[error\] Input model directory nonexistent/dir/ does not exist!" $output_dir_correct/corrected_reads.fasta.stderr | wc -l | awk '{ print $1 }')
set -e
if [[ $error_matched -ne "1" ]]; then
    echo "Dorado correct does not fail on non-existent input model path!"
    exit 1
fi

# Test decoupled mapping and inference stages.
#
output_dir_correct=${output_dir_correct_root}/test-04
mkdir -p ${output_dir_correct}
#
# Run the joined Dorado Correct pipeline (both mapping and inference stages) as a control.
$dorado_bin correct $data_dir/read_correction/reads.fq -v > $output_dir_correct/joined.fasta
samtools faidx $output_dir_correct/joined.fasta
#
# Run separate stages.
$dorado_bin correct $data_dir/read_correction/reads.fq -v --to-paf > $output_dir_correct/separate.paf
$dorado_bin correct $data_dir/read_correction/reads.fq -v --from-paf $output_dir_correct/separate.paf > $output_dir_correct/separate.fasta
samtools faidx $output_dir_correct/separate.fasta
#
# Evaluate. Check that the number of generated sequences and their length is the same.
cut -f 1-2 $output_dir_correct/joined.fasta.fai | sort > $output_dir_correct/joined.fasta.seqs
cut -f 1-2 $output_dir_correct/separate.fasta.fai | sort > $output_dir_correct/separate.fasta.seqs
set +e
result=$(diff $output_dir_correct/joined.fasta.seqs $output_dir_correct/separate.fasta.seqs | wc -l | awk '{ print $1 }')
set -e
if [[ $result -ne "0" ]]; then
    echo "Dorado Correct decoupled map/inference run does not generate the same output as the complete pipeline."
    diff $output_dir_correct/joined.fasta.seqs $output_dir_correct/separate.fasta.seqs
    exit 1
fi

# Test if nonexistent user-specified input PAF file will fail gracefully. This test _should_ fail, that's why we deactivate the -e.
#
output_dir_correct=${output_dir_correct_root}/test-05
mkdir -p ${output_dir_correct}
#
set +e
$dorado_bin correct $data_dir/read_correction/reads.fq -v --from-paf nonexistent.paf > $output_dir_correct/corrected_reads.fasta 2> $output_dir_correct/corrected_reads.fasta.stderr
error_matched=$(grep "\[error\] Input PAF path nonexistent.paf does not exist!" $output_dir_correct/corrected_reads.fasta.stderr | wc -l | awk '{ print $1 }')
set -e
if [[ $error_matched -ne "1" ]]; then
    echo "Dorado correct does not fail gracefully on non-existent input PAF file!"
    exit 1
fi

echo "Dorado correct tests done!"

rm -rf $output_dir
