#!/usr/bin/python
# Benchmark script for Dorado benchmarking.

import argparse
import logging
import os
from dataclasses import dataclass
import subprocess as sp
from typing import Optional, List
import sys
from pathlib import Path

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# TODO: Add RNA4 dataset
POD5_MAP = {
    "dna_kit14_400bps_5khz": os.environ["DNA_KIT14_400BPS_5KHZ_BENCHMARK_DATASET"]
}

# TODO: Add references
REF_MAP = {}


def run_cmd(cmd: List[str], stderr=sp.PIPE, stdout=sp.DEVNULL):
    logging.info("Running cmd: %s", " ".join(cmd))
    try:
        return sp.run(
            " ".join(cmd),
            shell=True,
            text=True,
            stderr=stderr,
            stdout=stdout,
            check=True,
        )
    except sp.CalledProcessError as e:
        print(e.stderr)
        raise e


@dataclass
class DoradoConfig:
    data_type: str
    model_variant: str
    duplex: Optional[bool] = False
    devices: Optional[str] = ""
    max_reads: Optional[int] = None
    mods: Optional[List[str]] = None
    reference: Optional[str] = None
    barcode_kit: Optional[str] = None

    @staticmethod
    def download_data(url: str) -> str:
        output_path = "data.pod5"
        if not os.path.exists(output_path):
            download_cmd = ["curl", "-LfS", "-o", output_path, url]
            run_cmd(download_cmd, stderr=None, stdout=None)
        return output_path

    def generate_dorado_cmd(self, dorado_bin: str):
        cmds = [dorado_bin]
        cmds.append("duplex" if self.duplex else "basecaller")
        cmds.append(self.model_variant)
        cmds.append(self.download_data(POD5_MAP[self.data_type]))
        if self.devices:
            cmds.extend(["-x", self.devices])
        if self.max_reads:
            cmds.extend(["--max-reads", str(self.max_reads)])
        if self.mods:
            cmds.append("--modified-bases")
            cmds.extend(self.mods)
        if self.reference:
            cmds.extend(["--ref", self.reference])
        if self.barcode_kit:
            cmds.extend(["--kit-name", self.barcode_kit])
        return cmds


# Track test configuration by name
TEST_MAP = {
    "dna_kit14_400bps_5khz_simplex_fast_osxarm_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="fast@v4.3.0",
        duplex=False,
        devices=None,
        max_reads=2000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_osxarm_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices=None,
        max_reads=500,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_sup_osxarm_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="sup@v4.3.0",
        duplex=False,
        devices=None,
        max_reads=100,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_osxarm_5mCG-5hmCG_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices=None,
        max_reads=500,
        mods=["5mCG_5hmCG"],
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_fast_cuda0_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="fast@v4.3.0",
        duplex=False,
        devices="cuda:0",
        max_reads=10000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_cuda0_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices="cuda:0",
        max_reads=5000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_sup_cuda0_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="sup@v4.3.0",
        duplex=False,
        devices="cuda:0",
        max_reads=1000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_cuda0_5mCG-5hmCG_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices="cuda:0",
        max_reads=5000,
        mods=["5mCG_5hmCG"],
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_fast_cudaall_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="fast@v4.3.0",
        duplex=False,
        devices="cuda:all",
        max_reads=10000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_cudaall_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices="cuda:all",
        max_reads=5000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_sup_cudaall_nomods_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="sup@v4.3.0",
        duplex=False,
        devices="cuda:all",
        max_reads=1000,
        mods=None,
        reference=None,
        barcode_kit=None,
    ),
    "dna_kit14_400bps_5khz_simplex_hac_cudaall_5mCG-5hmCG_noaln_nobarcode": DoradoConfig(
        data_type="dna_kit14_400bps_5khz",
        model_variant="hac@v4.3.0",
        duplex=False,
        devices="cuda:all",
        max_reads=5000,
        mods=["5mCG_5hmCG"],
        reference=None,
        barcode_kit=None,
    ),
}


def parse_speed(output):
    for line in output.splitlines():
        line = line.strip()
        if "Basecalled @ Samples/s" in line or "Basecalled @ Bases/s" in line:
            tokens = line.split(":")
            tput = float(tokens[-1].strip())
            return tput
    raise RuntimeError(f"Unable to parse speed from output \n{output}")


def log_speed(
    config: DoradoConfig,
    test_name: str,
    speed: float,
    output_dir: str,
    platform: str,
    gpu_type: str,
    sequencer: str,
):
    speed = int(speed)
    # Log speeds in two places
    logging.info("Dorado speed: %f", speed)
    path = Path(output_dir) / platform / sequencer / test_name
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "speed.csv", "w") as fh:
        fh.write(
            "type,duplex,model_variant,devices,max_reads,mods,reference,barcode_kit,platform,gpu_type,sequencer,speed\n"
        )
        fh.write(
            f"{config.data_type},{config.duplex},{config.model_variant},{config.devices},{config.max_reads},{config.mods},{config.reference},{config.barcode_kit},{platform},{gpu_type},{sequencer},{speed}\n"
        )


def parse_args():
    new_line = "\n"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-name",
        help=f"Name of the test to run. Options: \n{new_line.join(TEST_MAP.keys())}",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--gpu-type", help="Type of GPU being used", type=str, required=True
    )
    parser.add_argument(
        "--platform", help="Platform being run on", type=str, required=True
    )
    parser.add_argument(
        "--sequencer",
        help="Type of sequencing compute tower being run on",
        type=str,
        default="standalone",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for benchmark results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dorado-bin", help="Location of dorado binary", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tests = args.test_name
    for test_name in tests:
        if test_name not in TEST_MAP:
            new_line = "\n"
            raise RuntimeError(
                f"Test {test_name} not found. Please provide one of \n{new_line.join(TEST_MAP.keys())}"
            )

        logging.info("Running test %s", test_name)

        config = TEST_MAP[test_name]
        cmd = config.generate_dorado_cmd(args.dorado_bin)
        result = run_cmd(cmd)
        speed = parse_speed(result.stderr)
        log_speed(
            config,
            test_name,
            speed,
            args.output_dir,
            args.platform,
            args.gpu_type,
            args.sequencer,
        )
