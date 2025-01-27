#!/bin/bash

# Integration tests for Dorado Polish.
# Note: To disable these tests, set the following env variable: "NO_TEST_DORADO_POLISH=1".

set -ex
set -o pipefail

# Do nothing if this env variable is set.
if [[ "${NO_TEST_DORADO_POLISH}" == "1" ]]; then
    exit 0
fi

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
    output_dir_name=test_output_dc_$(echo $RANDOM | head -c 10)
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

# Download the model once.
MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
MODEL_DIR=${output_dir}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${output_dir}
fi

export DORADO_BIN
export TEST_DATA_DIR
export MODEL_DIR
python3 ${CRAM} --verbose ${TEST_DIR}/cram/polish/*.t
