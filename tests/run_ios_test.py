#!/usr/bin/env python3
#
# Helper to launch an app bundle on the currently booted device and check that
# it was successful.
#
import subprocess
from argparse import ArgumentParser
import typing as tp
import json


def get_devices() -> list[tuple[str, str]]:
    print("Getting device list")
    exe_args = [
        "xcrun",
        "simctl",
        "list",
        "-j",
    ]
    output = json.loads(subprocess.check_output(exe_args))
    devices = []
    for device_list in output["devices"].values():
        for device in device_list:
            state = device["state"]
            if device["isAvailable"] and (state == "Booted" or state == "Shutdown"):
                devices.append((device["name"], state))
    return devices


def setup_device(device: tp.Optional[str]) -> bool:
    needs_booting = True

    # Pick a device if one isn't provided.
    if device is None:
        devices = get_devices()
        for name, state in devices:
            if "iPad" in name and "Pro" in name:
                device = name
                needs_booting = state != "Booted"
                break
        if device is None:
            raise RuntimeError(f"Couldn't find appropriate device in: {devices}")
        print(f"Chose device '{device}'")

    if needs_booting:
        print(f"Booting up device: '{device}'")
        exe_args = [
            "xcrun",
            "simctl",
            "boot",
            device,
        ]
        return subprocess.call(exe_args) == 0
    else:
        print("Device is already booted")
        return True


def teardown_device(device: tp.Optional[str]) -> bool:
    if device is None:
        device = "all"
    print(f"Shutting down device: '{device}'")
    exe_args = [
        "xcrun",
        "simctl",
        "shutdown",
        device,
    ]
    return subprocess.call(exe_args) == 0


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
    parser = ArgumentParser()
    # Main operation.
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--setup", action="store_true")
    options.add_argument("--teardown", action="store_true")
    options.add_argument("--run", action="store_true")
    # Extra args.
    parser.add_argument("--app_path", type=str)
    parser.add_argument("--identifier", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    # Handle the args.
    success = False
    if args.setup:
        success = setup_device(args.device)
    elif args.teardown:
        success = teardown_device(args.device)
    elif args.run:
        # Assume a device is already running.
        if not args.device:
            args.device = "booted"
        # Uninstall the app first so that we can't accidentally run an old build.
        uninstall_app(args.device, args.identifier)
        install_app(args.device, args.app_path)
        success = run_test(args.device, args.identifier)
        uninstall_app(args.device, args.identifier)

    # Return the result to the ctest runner.
    if not success:
        exit(1)
