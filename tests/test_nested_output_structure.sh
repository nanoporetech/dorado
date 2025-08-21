#!/bin/bash

# Test expected output structure from the dorado binary execution.

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [model] [batch size]"
    exit 1
fi

test_dir=$(dirname $0)

# Main arguments
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model=${2:-"hac"}
batch=${3:-384}

# Test ion a tempdir to shorted filepaths which can cause issues on windows
TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t tmp)"
trap 'rm -rf "$TMPDIR"' EXIT

output_dir_name=test_output_nested_structure_${RANDOM}
output_dir=${TMPDIR}/${output_dir_name}
mkdir -p ${output_dir}

# Model storage
models_directory=${output_dir}/models
mkdir -p ${models_directory}

# Data source
data_dir=${test_dir}/data
demux_data=${data_dir}/barcode_demux/read_group_test
align_data=${data_dir}/pod5/dna_r10.4.1_e8.2_400bps_5khz
split_data=${data_dir}/single_split_read

# Common arguments
basic_args="-b ${batch} --models-directory ${models_directory} -v"
common_args=" basecaller ${model} ${demux_data} ${basic_args} "

check_structure() {
    set +x
    # check_structure {root_directory} {[expected_paths]}
    local root_directory="$1"
    local failed=0

    echo "Checking structure at ${root_directory}"

    # Shift $@ over to expected_paths
    shift
    local -a expected_paths=("$@")
    local expected_count=${#expected_paths[@]}

    # Check each expected file
    for relative_path in "${expected_paths[@]}"; do
        local full_path="${root_directory%/}/$relative_path"
        if [ ! -e "$full_path" ]; then
            echo "Error - Missing expected file: '$relative_path'"
            failed=1
        fi
    done

    # Count actual files (excluding directories) under root
    local actual_count
    actual_count=$(find "$root_directory" -type f | wc -l)

    # Check file count
    if [ "$actual_count" -ne "$expected_count" ]; then
        echo "Error - File count mismatch!"
        echo "  Expected: $expected_count"
        echo "  Found:    $actual_count"
        failed=1
    fi

    # Output filenames on error
    if [[ $failed -ne 0 ]]; then
        echo "Found: "
        find "$root_directory" -type f
    fi

    set -x
    return $failed
}

{
    # No demultiplexing
    echo "Testing basic nested output structure"
    dest="${output_dir}/basic_structure"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/bam_pass/PAO25751_pass_0d85015e_9bf5b3eb_0.bam"
    )

    $dorado_bin ${common_args} --output-dir ${dest}
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing without sample sheet into BAM
    echo "Testing nested output structure with demultiplexing into BAM"
    dest="${output_dir}/demux_structure_BAM"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/bam_pass/barcode01/PAO25751_pass_barcode01_0d85015e_9bf5b3eb_0.bam"
        "${core}/bam_pass/barcode04/PAO25751_pass_barcode04_0d85015e_9bf5b3eb_0.bam"
        "${core}/bam_pass/unclassified/PAO25751_pass_unclassified_0d85015e_9bf5b3eb_0.bam"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing without sample sheet into FASTQ
    echo "Testing nested output structure with demultiplexing into FASTQ"
    dest="${output_dir}/demux_structure_FASTQ"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/fastq_pass/barcode01/PAO25751_pass_barcode01_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/barcode04/PAO25751_pass_barcode04_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/unclassified/PAO25751_pass_unclassified_0d85015e_9bf5b3eb_0.fastq"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --emit-fastq
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing with sample sheet into SAM
    echo "Testing nested output structure with demultiplexing and sample sheet into SAM"
    dest="${output_dir}/demux_sample_sheet_structure_SAM"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/bam_pass/patient_id_1/PAO25751_pass_patient_id_1_0d85015e_9bf5b3eb_0.sam"
        "${core}/bam_pass/unclassified/PAO25751_pass_unclassified_0d85015e_9bf5b3eb_0.sam"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --emit-sam --sample-sheet ${data_dir}/barcode_demux/sample_sheet.csv
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing with sample sheet into FASTQ
    echo "Testing nested output structure with demultiplexing and sample sheet into FASTQ"
    dest="${output_dir}/demux_sample_sheet_structure"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/fastq_pass/patient_id_1/PAO25751_pass_patient_id_1_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/unclassified/PAO25751_pass_unclassified_0d85015e_9bf5b3eb_0.fastq"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --sample-sheet ${data_dir}/barcode_demux/sample_sheet.csv --emit-fastq
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing split reads 
    echo "Testing nested output structure with split reads"
    dest="${output_dir}/demux_split_read"
    core="E8p2p1_400bps/no_sample/20231121_1559_5B_PAS14411_76cd574f"
    expected=(
        "${core}/bam_pass/PAS14411_pass_76cd574f_f78e5963_0.bam"
    )

    $dorado_bin basecaller ${model} ${split_data} ${basic_args} --output-dir ${dest} 
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing aligned reads
    echo "Testing nested output structure with aligned reads"
    dest="${output_dir}/demux_aligned_reads"
    core="test/test/20231125_1913_test_TEST_4524e8b9"
    expected=(
        "${core}/bam_pass/TEST_pass_4524e8b9_test_0.bam"
        "${core}/bam_pass/TEST_pass_4524e8b9_test_0.bam.bai"
    )
    ref="${output_dir}/ref.fq"
    $dorado_bin basecaller ${model} ${align_data} ${basic_args} --emit-fastq > $ref
    $dorado_bin basecaller ${model} ${align_data} ${basic_args} --output-dir ${dest} --reference $ref
    check_structure ${dest} "${expected[@]}"
}

# rm -rf ${TMPDIR} // Trap will clean-up