#!/usr/bin/env python3
#
# Test model downloads and smoke them.
#

import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from fnmatch import fnmatch
import yaml

Job = Tuple[str, str, Optional[str]]


def smoke_test_model(
    dorado_bin: Path,
    test_dir: Path,
    models_dir: Path,
    simplex_model: str,
    mods_model: Optional[str],
):
    # Run a single read through the model.
    single_read = test_dir / "pod5" / "single_na24385.pod5"
    assert single_read.exists(), "input data does not exist. --data path is wrong"

    simplex_path = models_dir / simplex_model
    assert simplex_path.exists(), f"simplex model does not exist at {simplex_path}"

    args = [
        dorado_bin,
        "basecaller",
        # We don't care about sample rate mismatches.
        "--skip-model-compatibility-check",
        models_dir / simplex_model,
        single_read,
    ]

    if mods_model:
        mods_path = models_dir / mods_model
        assert mods_path.exists(), f"mods model does not exist at {mods_path}"
        args.extend(["--modified-bases-models", mods_path])

    subprocess.check_call(
        args,
        stdout=subprocess.DEVNULL,
    )


def unpack(listy_dict: Dict, key: str) -> Dict:
    """
    The downloader --list-structured uses lists of dicts which are uniquely
    keyed. Flatten this list into a dict for ease of use.
    """
    if not listy_dict:
        return {}

    out = {}
    for dct in listy_dict.get(key, []):
        out.update(dct)
    return out


def get_jobs(dorado_bin: Path) -> List[Job]:
    """
    Parse the output of dorado download --list-structured returning a list of all possible
    test jobs as [condition, simplex_model, mods_model]
    """
    jobs: List[Job] = []
    # Grab the list of models
    structured_models_result = subprocess.check_output(
        [dorado_bin, "download", "--list-structured"]
    )
    structured_models_dict = yaml.load(structured_models_result, Loader=yaml.Loader)

    for condition, cond_i in structured_models_dict.items():
        for sm, si in unpack(cond_i, "simplex_models").items():
            jobs.append((condition, sm, None))
            for mm, _ in unpack(si, "modified_models").items():
                jobs.append((condition, sm, mm))

    return jobs


def run_tests(
    dorado_bin: Path,
    test_dir: Path,
    models_dir: Optional[Path],
    simplex_glob: str,
    mods_glob: str,
) -> int:
    """Test all models and mods models - returns number of cases tested"""
    count_tested = 0
    # Setup a temp directory to do everything in
    with TemporaryDirectory() as temp_dir:
        if not models_dir:
            # Download the models.
            models_dir = Path(temp_dir)
            print(f"Downloading models into: `{models_dir}`")
            subprocess.check_call([dorado_bin, "download", "--directory", models_dir])

        for condition, simplex, mods in get_jobs(dorado_bin):
            if simplex_glob and not fnmatch(simplex, simplex_glob):
                print(
                    f"Skipped {condition=}, {simplex=}, {mods=} matching {simplex_glob=}"
                )
                continue

            if mods_glob and not fnmatch(mods, mods_glob):
                print(
                    f"Skipped {condition=}, {simplex=}, {mods=} matching {mods_glob=}"
                )
                continue

            print(f"Testing {condition=}, {simplex=}, {mods=}")
            smoke_test_model(
                dorado_bin=dorado_bin,
                test_dir=test_dir,
                models_dir=Path(models_dir),
                simplex_model=simplex,
                mods_model=mods,
            )
            count_tested += 1
    return count_tested


def main() -> int:
    """Main test application - returns exit code"""
    # Parse args.
    parser = ArgumentParser()
    parser.add_argument(
        "--exe",
        required=True,
        type=Path,
        help="path to dorado exe to test",
    )
    parser.add_argument("--data", required=True, type=Path, help="path to test data")
    parser.add_argument(
        "--models-directory",
        required=False,
        type=Path,
        default=None,
        help="optional path to ALL models skipping download",
    )

    parser.add_argument(
        "--simplex_glob",
        required=False,
        type=str,
        help="optional glob expr to filter simplex models",
    )
    parser.add_argument(
        "--mods_glob",
        required=False,
        type=str,
        help="optional glob expr to filter simplex models",
    )

    args = parser.parse_args()
    dorado_bin = Path(args.exe).resolve()
    test_dir = Path(args.data).resolve()
    models_dir = (
        Path(args.models_directory).resolve() if args.models_directory else None
    )
    count_tested = run_tests(
        dorado_bin=dorado_bin,
        test_dir=test_dir,
        models_dir=models_dir,
        simplex_glob=args.simplex_glob,
        mods_glob=args.mods_glob,
    )

    return 0 if count_tested > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
