#!/bin/bash
#
# Wrapper around bash to keep the cram test output around so we can check it if
# it fails in CI.
#
# cram calls the shell with "$0 -" then pipes the args in via stdin, so we only
# need to emulate that much.
#
if [[ $# -ne 1 ]] ; then
    echo "Unexpected args"
    exit 1
fi

# Put each test in its own folder.
test_name=$(basename ${TESTFILE})
mkdir -p ${OUTPUT_DIR}/${test_name}
cd ${OUTPUT_DIR}/${test_name}

# Start the bash process that'll actually do the work and feed it stdin.
exec 3> >(bash -)
while IFS=$'\n' read -r line ; do
    echo "$line" >&3
done

# Send it EOF and return its error code.
exec 3>&-
exit $?

