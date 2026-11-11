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
    output_dir_name=test_output_variant_${RANDOM}
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

SMALLVAR_MODEL_NAME_v520="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0"
SMALLVAR_MODEL_NAME_v600="dna_r10.4.1_e8.2_400bps_hac@v6.0.0_smallvar@v1.0"
POLISH_MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"

ALL_MODELS=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "smallvar" | sed -E 's/^[^"]*"([^"]*)".*/\1/')

# Download the models once.
for model in ${POLISH_MODEL_NAME} ${ALL_MODELS}
do
  model_dir=${output_dir}/${model}
  if [[ ! -d "${model_dir}" ]]; then
      ${DORADO_BIN} download --model "${model}" --models-directory ${output_dir}
  fi
done

export DORADO_BIN
export TEST_DATA_DIR
export TEST_DIR
export MODEL_ROOT_DIR=${output_dir}
export POLISH_MODEL=${MODEL_ROOT_DIR}/${POLISH_MODEL_NAME}
export SMALLVAR_MODEL_v600=${MODEL_ROOT_DIR}/${SMALLVAR_MODEL_NAME_v600}
export OUTPUT_DIR=${output_dir}
python3 \
    ${CRAM} \
    --verbose \
    --shell=${TEST_DIR}/cram/cram_shell_wrapper.sh \
    ${TEST_DIR}/cram/variant/*.t
