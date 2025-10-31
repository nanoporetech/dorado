#! /usr/bin/env python3

import argparse
import pathlib
from dataclasses import asdict, dataclass

from utilities import run_dorado

DEBUG_PRINT = True
SPEED_LOG_STRING = "> Basecalled @ Samples/s:"


@dataclass(slots=True)
class Benchmarker(object):
    input_file: str
    output_file: str
    device: str
    iterations: int
    run_for: int
    basecall_model: str
    binary_path: str | None
    modified_bases: str | None
    kit_name: str | None
    reference: str | None
    models_directory: str | None

    def pre_benchmarking(self) -> list[str]:
        for key, value in asdict(self).items():
            Benchmarker.debug_print(f"{key}: {value}")

        # Load the input file a few times to try and convince the system to cache it.
        input_path = pathlib.Path(self.input_file)
        for _ in range(3):
            with input_path.open("rb") as f:
                f.read()

        # Generate the dorado command-line arguments.
        if self.binary_path is None:
            dorado_exe = "dorado"
        else:
            dorado_exe = str(pathlib.Path(self.binary_path) / "dorado")
        args = [
            dorado_exe,
            "basecaller",
            self.basecall_model,
            self.input_file,
            "--device",
            self.device,
            "--run-for",
            str(self.run_for),
        ]
        if self.modified_bases is not None:
            args.extend(["--modified-bases", self.modified_bases])
        if self.kit_name is not None:
            args.extend(["--kit-name", self.kit_name])
        if self.reference is not None:
            args.extend(["--reference", self.reference])
        if self.models_directory is not None:
            args.extend(["--models-directory", self.models_directory])
        return args

    def run_benchmarking(self, dorado_args):
        summary_file = pathlib.Path(self.output_file)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        speed_data = []
        for iteration in range(self.iterations):
            print("Processing iteration", iteration)
            speed = self.run_iteration(dorado_args, iteration, summary_file)
            speed_data.append([iteration, speed])
        print("Writing summary")
        with summary_file.open("w") as summary:
            summary.write("iteration\tspeed\n")
            for entry in speed_data:
                summary.write("\t".join([str(val) for val in entry]))
                summary.write("\n")

    def run_iteration(
        self, dorado_args: list[str], iteration: int, summary_file: pathlib.Path
    ) -> int:
        bam_file = summary_file.parent / f"out_{iteration}.bam"
        log_file = summary_file.parent / f"benchmarking_{iteration}.log"
        timeout = self.run_for + 120  # Provide ample time for file-loading etc...
        with bam_file.open("wb") as outfile, log_file.open("w") as errfile:
            run_dorado(dorado_args, timeout, outfile=outfile, errfile=errfile)
        with log_file.open("r") as log_handle:
            log_data = log_handle.read()
            pos1 = log_data.find(SPEED_LOG_STRING)
        if pos1 == -1:
            print("Could not find performance data in dorado logging.")
            Benchmarker.debug_print(log_data)
            raise Exception("Benchmarking failed due to error processing output.")
        pos1 += len(SPEED_LOG_STRING) + 1
        pos2 = log_data.find("\n", pos1)
        if pos2 == -1:
            print("Error parsing dorado log for performance data.")
            Benchmarker.debug_print(log_data)
            raise Exception("Benchmarking failed due to error processing output.")
        speed_str = log_data[pos1:pos2]
        return int(float(speed_str))

    @staticmethod
    def debug_print(line):
        if DEBUG_PRINT:
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Dorado benchmarking script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        type=str,
        help="Input pod5 file to use for benchmarking.",
    )
    parser.add_argument(
        "--models_directory",
        default=None,
        type=str,
        help="Folder to save any downloaded models.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        type=str,
        help="Output bam file.",
    )
    parser.add_argument(
        "-d",
        "--dorado_bin_dir",
        default=None,
        help="Directory where dorado exes live. If left unset, the "
        "bin folder must be in the path.",
    )
    parser.add_argument(
        "--basecall_model",
        required=True,
        type=str,
        help="Model to use in benchmarking.",
    )
    parser.add_argument(
        "--modified_bases",
        default=None,
        type=str,
        help="Comma-separated list of base modifications to use in benchmarking.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        type=str,
        help="Optional reference index to use for alignment.",
    )
    parser.add_argument(
        "--kit_name",
        default=None,
        type=str,
        help="Optional barcode kit to use for barcoding.",
    )
    parser.add_argument(
        "-x",
        "--device",
        required=True,
        help="GPU device argument to pass to dorado.",
    )
    parser.add_argument(
        "--run_for",
        default=120,
        type=int,
        help="Number of seconds to run each benchmarking iteration for.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of iterations to run.",
    )
    args = parser.parse_args()
    benchmarker = Benchmarker(
        input_file=args.input_file,
        output_file=args.output_file,
        device=args.device,
        iterations=args.iterations,
        run_for=args.run_for,
        basecall_model=args.basecall_model,
        binary_path=args.dorado_bin_dir,
        modified_bases=args.modified_bases,
        kit_name=args.kit_name,
        reference=args.reference,
        models_directory=args.models_directory,
    )
    dorado_args = benchmarker.pre_benchmarking()
    benchmarker.run_benchmarking(dorado_args)
    print("Benchmarking complete")
