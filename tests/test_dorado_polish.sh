#!/bin/bash
#
# Integration tests for Dorado Polish.
#

# Common setup.
fallback_output_dir="test_output_polish"
source "$(dirname $0)/test_cram_common.sh"

SMALLVAR_MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0"
POLISH_MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"

# Download the models once.
ALL_MODELS=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "polish" | sed -E 's/^[^"]*"([^"]*)".*/\1/' | grep -v "polish_models" | sort -u)
for model in ${SMALLVAR_MODEL_NAME} ${ALL_MODELS}
do
  model_dir=${output_dir}/${model}
  if [[ ! -d "${model_dir}" ]]; then
      ${DORADO_BIN} download --model "${model}" --models-directory ${output_dir}
  fi
done

# Run the tests.
export MODEL_ROOT_DIR=${output_dir}
export POLISH_MODEL=${MODEL_ROOT_DIR}/${POLISH_MODEL_NAME}
export SMALLVAR_MODEL=${MODEL_ROOT_DIR}/${SMALLVAR_MODEL_NAME}
run_cram_tests "polish"
