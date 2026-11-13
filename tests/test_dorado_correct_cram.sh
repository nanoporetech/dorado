#!/bin/bash
#
# Integration tests for Dorado Correct.
#

# Common setup.
fallback_output_dir="test_output_correct_cram"
source "$(dirname $0)/test_cram_common.sh"

# Download the model once.
MODEL_NAME="herro-v1.1"
MODEL_DIR=${OUTPUT_DIR}/${MODEL_NAME}
if [[ ! -d "${MODEL_DIR}" ]]; then
    ${DORADO_BIN} download --model "${MODEL_NAME}" --models-directory ${OUTPUT_DIR}
fi

# Run the tests.
export MODEL_DIR
export MODEL_NAME
export MODEL_ROOT_DIR=${OUTPUT_DIR}
run_cram_tests "correct"
