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
if [[ "${OUT_DIR}" == "" ]]; then
    output_dir_name=${fallback_output_dir}_${RANDOM}
    OUTPUT_DIR=${TEST_DIR}/${output_dir_name}
else
    OUTPUT_DIR=$(cd "${OUT_DIR}"; pwd -P)
fi
mkdir -p ${OUTPUT_DIR}

# Download the Cram package.
python3 -m venv "${TEST_DIR}"/.cram_venv
source "${TEST_DIR}"/.cram_venv/*/activate
pip install "cram==0.6"

# Once the calling script has setup/cached what it needs it should call this to run the tests.
function run_cram_tests() {
    tests_to_run=("$@")

    # Shard the tests.
    if [[ -n $SHARD_COUNT ]] && [[ -n $SHARD_IDX ]]; then
        shard_size=$(( ( ${#tests_to_run[@]} + ${SHARD_COUNT} - 1) / ${SHARD_COUNT} ))
        shard_start=$(( ${shard_size} * ${SHARD_IDX} ))
        tests_to_run=( ${tests_to_run[@]:${shard_start}:${shard_size}} )
    fi

    export DORADO_BIN
    export TEST_DATA_DIR
    export TEST_DIR
    export OUTPUT_DIR
    cram \
        --verbose \
        --shell=${TEST_DIR}/cram/cram_shell_wrapper.sh \
        ${tests_to_run[@]}
}
