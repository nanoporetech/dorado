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

# Grab the tests.
test_cases=(  $(ls "${TEST_DIR}"/cram/correct/cram-correct-blocks-??.t     | sort) )
test_cases+=( $(ls "${TEST_DIR}"/cram/correct/cram-correct-index-size-??.t | sort) )
test_cases+=( $(ls "${TEST_DIR}"/cram/correct/cram-correct-resume-??.t     | sort) )

# Run the tests.
export MODEL_DIR
export MODEL_NAME
export MODEL_ROOT_DIR=${OUTPUT_DIR}
run_cram_tests "${test_cases[@]}"
