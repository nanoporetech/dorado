#!/bin/bash
#
# Utils for running cram tests.
#

if [[ -z ${fallback_output_dir} ]] ; then
    echo "This script isn't meant to be called directly. Run the test_*.sh scripts instead."
    exit 1
fi

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [<out_dir>]"
    exit 1
fi

if [[ $# -eq 2 ]]; then
    mkdir -p $2
fi

# CLI options.
IN_DORADO_BIN="$1"
OUT_DIR="$2"

TEST_DIR=$(cd "$(dirname $0)"; pwd -P)
TEST_DATA_DIR=${TEST_DIR}/data
DORADO_BIN=$(cd "$(dirname ${IN_DORADO_BIN})"; pwd -P)/$(basename ${IN_DORADO_BIN})

# Output directory. Either user specified or generated.
OUTPUT_DIR=$(cd "${OUT_DIR}"; pwd -P)
if [[ "${OUT_DIR}" == "" ]]; then
    output_dir_name=${fallback_output_dir}_${RANDOM}
    OUTPUT_DIR=${TEST_DIR}/${output_dir_name}
fi

mkdir -p ${OUTPUT_DIR}

# Download the Cram package.
pushd ${OUTPUT_DIR}
    if [ ! -d cram-0.6 ]; then
        curl -o cram-0.6.tar.gz https://bitheap.org/cram/cram-0.6.tar.gz
        tar -xzf cram-0.6.tar.gz
    fi
    CRAM=$(pwd)/cram-0.6/cram.py
popd

# Once the calling script has setup/cached what it needs it should call this to run the tests.
function run_cram_tests() {
    cram_test_folder=$1

    # Shard the tests.
    tests_to_run=( $(ls ${TEST_DIR}/cram/${cram_test_folder}/*.t) )
    if [[ -n $SHARD_COUNT ]] && [[ -n $SHARD_IDX ]]; then
        shard_size=$(( ( ${#tests_to_run[@]} + ${SHARD_COUNT} - 1) / ${SHARD_COUNT} ))
        shard_start=$(( ${shard_size} * ${SHARD_IDX} ))
        tests_to_run=( ${tests_to_run[@]:${shard_start}:${shard_size}} )
    fi

    export DORADO_BIN
    export TEST_DATA_DIR
    export TEST_DIR
    export OUTPUT_DIR
    python3 \
        ${CRAM} \
        --verbose \
        --shell=${TEST_DIR}/cram/cram_shell_wrapper.sh \
        ${tests_to_run[@]}
}
