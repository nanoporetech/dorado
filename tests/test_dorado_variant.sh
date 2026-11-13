#!/bin/bash
#
# Integration tests for Dorado Polish.
#

# Common setup.
fallback_output_dir="test_output_variant"
source "$(dirname $0)/test_cram_common.sh"

SMALLVAR_MODEL_NAME_v520="dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0"
SMALLVAR_MODEL_NAME_v600="dna_r10.4.1_e8.2_400bps_hac@v6.0.0_smallvar@v1.0"
POLISH_MODEL_NAME="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"

# Download the models once.
ALL_MODELS=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "smallvar" | sed -E 's/^[^"]*"([^"]*)".*/\1/')
for model in ${POLISH_MODEL_NAME} ${ALL_MODELS}
do
  model_dir=${OUTPUT_DIR}/${model}
  if [[ ! -d "${model_dir}" ]]; then
      ${DORADO_BIN} download --model "${model}" --models-directory ${OUTPUT_DIR}
  fi
done

# Run the tests.
export MODEL_ROOT_DIR=${OUTPUT_DIR}
export POLISH_MODEL=${MODEL_ROOT_DIR}/${POLISH_MODEL_NAME}
export SMALLVAR_MODEL_v600=${MODEL_ROOT_DIR}/${SMALLVAR_MODEL_NAME_v600}
run_cram_tests "variant"
