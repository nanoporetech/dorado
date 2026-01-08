#!/bin/bash

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable>"
    exit 1
fi

test_dir=$(dirname $0)

# Main arguments
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)

# Test on a tempdir to shorted filepaths which can cause issues on windows
TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t tmp)"
trap 'rm -rf "$TMPDIR"' EXIT

# Create test output directory
output_dir_name=test_model_resolution_${RANDOM}
output_dir=${TMPDIR}/${output_dir_name}
mkdir -p ${output_dir}

# Data source
data_dir=$(realpath ${test_dir}/data)
example_data=${data_dir}/pod5/dna_r10.4.1_e8.2_400bps_5khz

# Common arguments
common_args=" --recursive --max-reads 1 -b 128 -v "
models_directory_arg="--models-directory ${models_directory}"

RUN_TESTS_DOWNLOADER_NAMED=1
RUN_TESTS_DOWNLOADER_VARIANT=1
RUN_TESTS_BASECALLER=1
RUN_TESTS_BASECALLER_PATH=1

check_exists() {
    set +x
    local search_directory="$1"

    # Shift $@ over to expected_paths
    shift
    local -a expected_paths=("$@")
    # Check each expected file
    for expected in "${expected_paths[@]}"; do
        echo "Checking ${search_directory} for ${expected}"
        local full_path="${search_directory%/}/$expected"
        
        if [ ! -e "$full_path" ]; then 
            echo "Error - Missing expected file: ${expected}"  
            exit 1
        fi
    done;

    set -x
    return 0
}

title() {
    set +x 
    echo ""
    echo ""
    echo "====================================================================="
    echo $1
    echo "====================================================================="
    set -x
}

# Testing dorado downloader
if [ $RUN_TESTS_DOWNLOADER_NAMED -eq 1 ]; then 
{
    title "Test we can download a named model"
    simplex_name="rna004_130bps_fast@v5.2.0"
    local_dir="${output_dir}/download_simplex"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${simplex_name}
        check_exists ${local_dir} ${simplex_name}
    popd
}
{
    title "Test we can download a named modbase model"
    modbase_name="rna004_130bps_hac@v5.2.0_m5C@v1"
    local_dir="${output_dir}/download_modbase"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${modbase_name}
        check_exists ${local_dir} ${modbase_name}
    popd
}
{
    title "Test we can download a named stereo model"
    stereo_name="dna_r10.4.1_e8.2_5khz_stereo@v1.4"
    local_dir="${output_dir}/download_stereo"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${stereo_name}
        check_exists ${local_dir} ${stereo_name}
    popd
}
{
    title "Test we do not re-download a named model"
    model_name="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    local_dir="${output_dir}/download_repeat"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${model_name} -vv 2> first_download.stderr.txt
        cat first_download.stderr.txt >&2
        ${dorado_bin} download --model ${model_name} -vv 2> repeat_download.stderr.txt
        cat repeat_download.stderr.txt >&2
        if ! grep -q "found existing model '${model_name}'" repeat_download.stderr.txt; then
            cat repeat_download.stderr.txt
            echo "Model download was not skipped"
            exit 1
        fi
    popd
}
fi # RUN_TESTS_DOWNLOADER_NAMED

if [ $RUN_TESTS_DOWNLOADER_VARIANT -eq 1 ]; then 
{
    title "Test we can download using a model complex"
    local_dir="${output_dir}/download_complex"
    complex="fast@v5.2"
    expected="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${complex} --data ${example_data}
        check_exists ${local_dir} ${expected}
    popd
}
{
    title "Test we can download using a model complex with modbases"
    local_dir="${output_dir}/download_modbase_complex2"
    complex="hac@v5.2,5mC_5hmC@v2,6mA@v1"
    expected=(
        "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
        "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
        "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_6mA@v1"
    )
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} download --model ${complex} --data ${example_data}
        check_exists ${local_dir} "${expected[@]}"
    popd
}
fi # RUN_TESTS_DOWNLOADER_VARIANT


if [ $RUN_TESTS_BASECALLER -eq 1 ]; then 
{
    title "Test we can auto download using a model complex which is then cleaned up"
    local_dir="${output_dir}/basecaller_variant_complex_temporary"
    complex="fast@v5.2"
    expected="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        # Check log for confirmation that expected model was downloaded
        if ! grep -q "downloading ${expected}" basecaller.stderr.txt; then
            cat basecaller.stderr.txt
            echo "Model download error"
            exit 1
        fi

        # Check that the temporary model was deleted
        if [ -e "${local_dir}/${expected}" ]; then 
            echo "Expected temporary model '${expected}' to be deleted"
            exit 1
        fi
    popd
}
{
    title "Test we can auto download using a model complex into models-directory"
    local_dir="${output_dir}/basecaller_variant_complex_persistent"
    complex="fast@v5.2"
    expected="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} --models-directory . > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        # Check log for confirmation that expected model was downloaded
        if ! grep -q "downloading ${expected}" basecaller.stderr.txt; then
            cat basecaller.stderr.txt
            echo "Model download error"
            exit 1
        fi

        # Check that the model was NOT deleted
        check_exists ${local_dir} ${expected}
    popd
}
{
    title "Test we can auto download model by name"
    local_dir="${output_dir}/basecaller_named_complex_persistent"
    complex="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    expected="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} --models-directory . > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        # Check log for confirmation that expected model was downloaded
        if ! grep -q "downloading ${expected}" basecaller.stderr.txt; then
            cat basecaller.stderr.txt
            echo "Model download error"
            exit 1
        fi

        check_exists ${local_dir} ${expected}
    popd
}
{
    title "Test we can auto download a modbase model and it's parent simplex model modbase model name"
    local_dir="${output_dir}/basecaller_named_modbase"
    complex="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
    expected=("dna_r10.4.1_e8.2_400bps_hac@v5.2.0" "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2")
    mkdir -p ${local_dir}
    pushd ${local_dir}
        ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} --models-directory . > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        check_exists ${local_dir} "${expected[@]}"
    popd
}
{
    title "Test we can reuse named modbase model and specify modbasecalling via complex"
    local_dir="${output_dir}/basecaller_named_modbase_reuse"
    complex="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
    expected=("dna_r10.4.1_e8.2_400bps_hac@v5.2.0" "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2")
    mkdir -p ${local_dir}
    pushd ${local_dir}
        mkdir -p models
        ${dorado_bin} download --model hac@v5.2.0,5mC_5hmC@v2 --data ${example_data} --models-directory models
        ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} -vv --models-directory models  > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        # Check log for confirmation that expected models were found
        for ex in "${expected[@]}"; do
            echo "Checking ${search_directory} for ${ex}"
            if ! grep -q "found model '${ex}'" basecaller.stderr.txt; then
                cat basecaller.stderr.txt
                echo "Model reuse error"
                exit 1
            fi
        done;
        
        check_exists ${local_dir}/models "${expected[@]}"
    popd
}
{
    title "Check mixing named modbase is not permitted with other arguments"
    local_dir="${output_dir}/basecaller_named_modbase"
    complex="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        if ${dorado_bin} basecaller ${complex} ${example_data} ${common_args} --modified-bases 5mC_5hmC > /dev/null 2> basecaller.stderr.txt; then
            cat basecaller.stderr.txt >&2
            echo "Expected dorado to error"
            exit 1
        fi
        cat basecaller.stderr.txt >&2
    popd
}
fi # RUN_TESTS_BASECALLER

if [ $RUN_TESTS_BASECALLER_PATH -eq 1 ]; then 
{
    title "Test we can use a model by path"
    local_dir="${output_dir}/basecaller_path"
    name="dna_r10.4.1_e8.2_400bps_fast@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        mkdir -p models
        ${dorado_bin} download --model ${name} --models-directory models
        ${dorado_bin} basecaller ./models/${name} ${example_data} ${common_args} > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2
    popd
}
{
    title "Test we can specify a model by path and auto download a modbase model via --modified-bases"
    local_dir="${output_dir}/basecaller_path_modified_bases"
    name="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        mkdir -p models
        ${dorado_bin} download --model ${name} --models-directory models
        ${dorado_bin} basecaller ./models/${name} ${example_data} ${common_args} --modified-bases "5mC_5hmC" > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2

        # Check log for confirmation that expected model was downloaded
        if ! grep -q "downloading dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2" basecaller.stderr.txt; then
            cat basecaller.stderr.txt
            echo "Model download error"
            exit 1
        fi
    popd
}
{
    title "Test we can specify a model by path and --modified-bases-models by path"
    local_dir="${output_dir}/basecaller_path_modified_paths"
    name="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    mod="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        mkdir -p models
        ${dorado_bin} download --model ${name} --models-directory models
        ${dorado_bin} download --model ${mod}  --models-directory models
        ${dorado_bin} basecaller ./models/${name} ${example_data} ${common_args} -vv --modified-bases-models ./models/${mod} > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2
    popd
}
{
    title "Test we can specify a model by path and --modified-bases-models by name"
    local_dir="${output_dir}/basecaller_path_modified_name"
    name="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    mod="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"
    mkdir -p ${local_dir}
    pushd ${local_dir}
        mkdir -p models
        ${dorado_bin} download --model ${name} --models-directory models
        ${dorado_bin} download --model ${mod}  --models-directory models
        ${dorado_bin} basecaller ./models/${name} ${example_data} ${common_args} -vv --modified-bases-models ${mod} > /dev/null 2> basecaller.stderr.txt
        cat basecaller.stderr.txt >&2
        # Check log for confirmation that expected model was downloaded
        if ! grep -q "downloading dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2" basecaller.stderr.txt; then
            echo "Model download error"
            exit 1
        fi
    popd
}
fi # RUN_TESTS_BASECALLER_PATH

title "DONE"

# rm -rf ${TMPDIR} // Trap will clean-up
