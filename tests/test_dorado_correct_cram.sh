#!/bin/bash

# Integration tests for Dorado Correct.
# Note: To disable these tests, set the following env variable: "NO_TEST_DORADO_CORRECT=1".

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [<out_dir>]"
    exit 1
fi

# CLI options.
IN_DORADO_BIN="$1"
OUT_DIR="$2"

# Do nothing if this env variable is set.
if [[ "${NO_TEST_DORADO_CORRECT}" == "1" ]]; then
    exit 0
fi

TEST_DIR=$(cd "$(dirname $0)"; pwd -P)
TEST_DATA_DIR=${TEST_DIR}/data
DORADO_BIN=$(cd "$(dirname ${IN_DORADO_BIN})"; pwd -P)/$(basename ${IN_DORADO_BIN})

# Output directory. Either user specified or generated.
output_dir=$(cd "${OUT_DIR}"; pwd -P)
if [[ "${OUT_DIR}" == "" ]]; then
    output_dir_name=test_output_dc_$(echo $RANDOM | head -c 10)
    output_dir=${TEST_DIR}/${output_dir_name}
fi

mkdir -p ${output_dir}

# Download the Cram package.
pushd ${output_dir}
curl -o cram-0.6.tar.gz https://bitheap.org/cram/cram-0.6.tar.gz
tar -xzf cram-0.6.tar.gz
CRAM=$(pwd)/cram-0.6/cram.py
popd

# Set the model directory for the tests.
MODEL_DIR=${output_dir}/herro-v1

# Download the model once.
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "herro-v1" --models-directory ${output_dir}
fi

export DORADO_BIN
export TEST_DATA_DIR
export MODEL_DIR
python3 ${CRAM} --verbose ${TEST_DIR}/cram/*.t
