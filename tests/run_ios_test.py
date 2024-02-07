#!/usr/bin/env python3
#
# Helper to launch an app bundle on the currently booted device and check that
# it was successful.
#
import subprocess
from argparse import ArgumentParser

# Parse args.
parser = ArgumentParser()
parser.add_argument("--identifier", required=True, type=str)
args = parser.parse_args()

# Run the test.
exe_args = [
    "xcrun",
    "simctl",
    "launch",
    "--console",  # Note: this merges stderr into stdout
    "--terminate-running-process",
    "booted",
    args.identifier,
]
success_string = "ALL TESTS RAN SUCCESSFULLY"
seen_success = False
with subprocess.Popen(exe_args, stdout=subprocess.PIPE) as proc:
    # Print messages as we go otherwise we won't be able to diagnose hangs.
    for line in iter(proc.stdout.readline, b""):
        line = line.decode(errors="replace")
        print(line)
        if success_string in line:
            seen_success = True

# Return the result to the ctest runner.
if not seen_success:
    print("No success message spotted in output")
    exit(1)
