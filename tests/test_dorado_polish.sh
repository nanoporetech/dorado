#!/bin/bash

# Integration tests for Dorado Polish.

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
    output_dir_name=test_output_polish_${RANDOM}
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
SMALLVAR_MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0"
SMALLVAR_MODEL_DIR=${output_dir}/${SMALLVAR_MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${SMALLVAR_MODEL_NAME}" --models-directory ${output_dir}
fi

# Download the model once.
MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish"
MODEL_DIR=${MODEL_ROOT_DIR}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${output_dir}
fi

# Download the model once.
MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
MODEL_DIR=${MODEL_ROOT_DIR}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${output_dir}
fi

# Download the model once.
MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
MODEL_DIR=${MODEL_ROOT_DIR}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${output_dir}
fi

# Download the model once.
MODEL_NAME_v52="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl_mv"
MODEL_DIR_v52=${MODEL_ROOT_DIR}/${MODEL_NAME_v52}
if [[ ! -d "${MODEL_DIR_v52}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME_v52}" --models-directory ${output_dir}
fi

export DORADO_BIN
export TEST_DATA_DIR
export MODEL_NAME
export MODEL_DIR
export MODEL_ROOT_DIR
export SMALLVAR_MODEL_NAME
export SMALLVAR_MODEL_DIR
export OUTPUT_DIR=${output_dir}
python3 \
    ${CRAM} \
    --verbose \
    --shell=${TEST_DIR}/cram/cram_shell_wrapper.sh \
    ${TEST_DIR}/cram/polish/cram-polish-??-*.t \
    ${TEST_DIR}/cram/polish/cram-polish-models*.t
