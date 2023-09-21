#!/usr/bin/env python3
#
# Test model downloads and smoke them.
#
import subprocess
import yaml
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

# Parse args.
parser = ArgumentParser()
parser.add_argument("--exe", required=True, type=str, help=".")
parser.add_argument("--data", required=True, type=str, help=".")
args = parser.parse_args()
dorado_bin = Path(args.exe)
test_dir = Path(args.data)

# Setup a temp directory to do everything in
with TemporaryDirectory() as models_dir:
    # Download the models.
    print("Downloading models")
    subprocess.check_call([dorado_bin, "download", "--directory", models_dir])

    # Grab the list of models
    all_models = subprocess.check_output([dorado_bin, "download", "--list-yaml"])
    all_models = yaml.load(all_models, Loader=yaml.Loader)

    # Run a single read through each model.
    single_read = test_dir / "pod5" / "single_na24385.pod5"
    for model in all_models["simplex models"]:
        print(f"Testing {model}")
        subprocess.check_call(
            [
                dorado_bin,
                "basecaller",
                # We don't care about sample rate mismatches.
                "--skip-model-compatibility-check",
                Path(models_dir) / model,
                single_read,
            ],
            stdout=subprocess.DEVNULL,
        )
