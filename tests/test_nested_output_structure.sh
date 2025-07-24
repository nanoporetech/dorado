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

# Test output
output_dir_name=test_output_nested_structure_${RANDOM}
output_dir=${test_dir}/${output_dir_name}
mkdir -p ${output_dir}

# Model storage
models_directory=${output_dir}/models
mkdir -p ${models_directory}

# Data source
data_dir=${test_dir}/data
demux_data=${data_dir}/barcode_demux/read_group_test

# Common arguments
common_args=" basecaller ${model} ${demux_data} -b ${batch} --models-directory ${models_directory} -v "

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
        "${core}/bam_pass/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.bam"
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
        "${core}/bam_pass/barcode01/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.bam"
        "${core}/bam_pass/barcode04/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.bam"
        "${core}/bam_pass/unclassified/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.bam"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing without sample sheet into SAM
    echo "Testing nested output structure with demultiplexing into SAM"
    dest="${output_dir}/demux_structure_SAM"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/bam_pass/barcode01/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.sam"
        "${core}/bam_pass/barcode04/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.sam"
        "${core}/bam_pass/barcode68/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.sam"
        "${core}/bam_pass/unclassified/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.sam"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --emit-sam
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing without sample sheet into FASTQ
    echo "Testing nested output structure with demultiplexing into FASTQ"
    dest="${output_dir}/demux_structure_FASTQ"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/fastq_pass/barcode01/PAO25751_fastq_pass_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/barcode04/PAO25751_fastq_pass_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/unclassified/PAO25751_fastq_pass_0d85015e_9bf5b3eb_0.fastq"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --emit-fastq
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing with sample sheet into BAM
    echo "Testing nested output structure with demultiplexing and sample sheet into BAM"
    dest="${output_dir}/demux_sample_sheet_structure_BAM"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/bam_pass/barcode01/PAO25751_bam_pass_patient_id_1_0d85015e_9bf5b3eb_0.bam"
        "${core}/bam_pass/unclassified/PAO25751_bam_pass_0d85015e_9bf5b3eb_0.bam"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --sample-sheet ${data_dir}/barcode_demux/sample_sheet.csv
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing with sample sheet into FASTQ
    echo "Testing nested output structure with demultiplexing and sample sheet into FASTQ"
    dest="${output_dir}/demux_sample_sheet_structure"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/fastq_pass/barcode01/PAO25751_fastq_pass_patient_id_1_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/unclassified/PAO25751_fastq_pass_0d85015e_9bf5b3eb_0.fastq"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --sample-sheet ${data_dir}/barcode_demux/sample_sheet.csv --emit-fastq
    check_structure ${dest} "${expected[@]}"
}

{
    # Demultiplexing with sample sheet into FASTQ
    echo "Testing nested output structure with demultiplexing and sample sheet into FASTQ"
    dest="${output_dir}/demux_sample_sheet_structure"
    core="no_sample/20230807_1018_2H_PAO25751_0d85015e"
    expected=(
        "${core}/fastq_pass/barcode01/PAO25751_fastq_pass_patient_id_1_0d85015e_9bf5b3eb_0.fastq"
        "${core}/fastq_pass/unclassified/PAO25751_fastq_pass_0d85015e_9bf5b3eb_0.fastq"
    )

    $dorado_bin ${common_args} --output-dir ${dest} --kit-name SQK-RBK114-96 --sample-sheet ${data_dir}/barcode_demux/sample_sheet.csv --emit-fastq
    check_structure ${dest} "${expected[@]}"
}

rm -rf ${output_dir}