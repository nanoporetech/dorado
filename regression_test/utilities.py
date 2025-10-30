import os
import pathlib
import subprocess
import typing

DEBUG = os.getenv("BUILD_TYPE", "Release").upper() == "DEBUG"


def run_dorado(
    cmd_args: list,
    timeout: int,
    outfile: typing.IO | None = None,
    errfile: typing.IO | None = None,
):
    print("Dorado command line: ", " ".join(cmd_args))
    out = subprocess.PIPE if outfile is None else outfile
    err = subprocess.PIPE if errfile is None else errfile
    result = subprocess.run(
        cmd_args, timeout=timeout, text=True, stdout=out, stderr=err
    )
    if result.returncode != 0:
        raise Exception(
            f"Error running {cmd_args}: returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def make_summary(input_file: pathlib.Path, save_filename: str, timeout: int):
    save_file = input_file.parent / save_filename
    args = ["doradod"] if DEBUG else ["dorado"]
    args.extend(
        [
            "summary",
            str(input_file),
        ]
    )
    print("Summary command line: ", " ".join(args))
    with save_file.open("w") as summary_out:
        result = subprocess.run(
            args, timeout=timeout, text=True, stdout=summary_out, stderr=subprocess.PIPE
        )
    if result.returncode != 0:
        raise Exception(
            f"Error running {args}: returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
