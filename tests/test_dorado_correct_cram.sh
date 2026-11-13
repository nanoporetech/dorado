#!/bin/bash

# Integration tests for Dorado Correct.

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
output_dir=$(cd "${OUT_DIR}"; pwd -P)
if [[ "${OUT_DIR}" == "" ]]; then
    output_dir_name=test_output_correct_cram_${RANDOM}
    output_dir=${TEST_DIR}/${output_dir_name}
fi

mkdir -p ${output_dir}

# Download the Cram package.
pushd ${output_dir}
if [ ! -d cram-0.6 ]; then
    curl -o cram-0.6.tar.gz https://bitheap.org/cram/cram-0.6.tar.gz
    tar -xzf cram-0.6.tar.gz
fi
CRAM=$(pwd)/cram-0.6/cram.py
popd

MODEL_ROOT_DIR=${output_dir}

# Download the model once.
MODEL_NAME="herro-v1.1"
MODEL_DIR=${MODEL_ROOT_DIR}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${MODEL_ROOT_DIR}
fi

# Shard the tests.
tests_to_run=( $(ls ${TEST_DIR}/cram/correct/*.t) )
if [[ -n $SHARD_COUNT ]] && [[ -n $SHARD_IDX ]]
    shard_size=$(( ( ${#tests_to_run[@]} + ${SHARD_COUNT} - 1) / ${SHARD_COUNT} ))
    shard_start=$(( ${shard_size} * ${SHARD_IDX} ))
    tests_to_run=${tests_to_run[@]:${shard_start}:${shard_size}}
fi

export DORADO_BIN
export TEST_DATA_DIR
export MODEL_DIR
export MODEL_NAME
export MODEL_ROOT_DIR
export OUTPUT_DIR=${output_dir}
python3 \
    ${CRAM} \
    --verbose \
    --shell=${TEST_DIR}/cram/cram_shell_wrapper.sh \
    ${tests_to_run[@]}
