#!/usr/bin/env bash

set -euo pipefail

if grep -n '^RUN_TESTS_.*=0' tests/*.sh; then
    echo "Found disabled RUN_TESTS_ in test scripts. Set them to '=1'."
    exit 1
fi
