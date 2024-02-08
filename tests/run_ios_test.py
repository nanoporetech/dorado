#!/usr/bin/env python3
#
# Helper to launch an app bundle on the currently booted device and check that
# it was successful.
#
import subprocess
from argparse import ArgumentParser


def install_app(device: str, app_path: str) -> bool:
    print(f"Installing {app_path} to {device}")
    exe_args = [
        "xcrun",
        "simctl",
        "install",
        device,
        app_path,
    ]
    return subprocess.call(exe_args) == 0


def uninstall_app(device: str, identifier: str) -> bool:
    print(f"Uninstalling {identifier} from {device}")
    exe_args = [
        "xcrun",
        "simctl",
        "uninstall",
        device,
        identifier,
    ]
    return subprocess.call(exe_args) == 0


def run_test(device: str, identifier: str) -> bool:
    print(f"Running {identifier} on {device}")
    exe_args = [
        "xcrun",
        "simctl",
        "launch",
        "--console",  # Note: this merges stderr into stdout
        "--terminate-running-process",
        device,
        identifier,
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
    if not seen_success:
        print("No success message spotted in output")
    return seen_success


if __name__ == "__main__":
    # Parse args.
    parser = ArgumentParser()
    parser.add_argument("--app_path", required=True, type=str)
    parser.add_argument("--identifier", required=True, type=str)
    parser.add_argument("--device", default="booted", type=str)
    args = parser.parse_args()

    # Uninstall the app first so that we can't accidentally run an old build.
    uninstall_app(args.device, args.identifier)
    install_app(args.device, args.app_path)
    success = run_test(args.device, args.identifier)
    uninstall_app(args.device, args.identifier)

    # Return the result to the ctest runner.
    if not success:
        exit(1)
