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
dorado_bin = Path(args.exe).resolve()
test_dir = Path(args.data).resolve()


def smoke_test_model(model: Path, extra_args=[]):
    # Run a single read through the model.
    single_read = test_dir / "pod5" / "single_na24385.pod5"
    args = [
        dorado_bin,
        "basecaller",
        # We don't care about sample rate mismatches.
        "--skip-model-compatibility-check",
        model,
        single_read,
    ]
    args.extend(extra_args)
    subprocess.check_call(
        args,
        stdout=subprocess.DEVNULL,
    )


# Setup a temp directory to do everything in
with TemporaryDirectory() as models_dir:
    # Download the models.
    print("Downloading models")
    subprocess.check_call([dorado_bin, "download", "--directory", models_dir])

    # Grab the list of models
    all_models = subprocess.check_output([dorado_bin, "download", "--list-yaml"])
    all_models = yaml.load(all_models, Loader=yaml.Loader)
    simplex_models = all_models["simplex models"]
    modbase_models = all_models["modification models"]

    # Smoke test each set.
    for simplex_model in simplex_models:
        print(f"Testing {simplex_model}")
        smoke_test_model(Path(models_dir) / simplex_model)

    for modbase_model in modbase_models:
        print(f"Testing {modbase_model}")
        # Extract the simplex model and modbases: <simplex>_<modbase>@<version
        for simplex_model in simplex_models:
            if modbase_model.startswith(simplex_model + "_"):
                modbase = modbase_model[len(simplex_model) + 1 :]
                modbase = modbase.split("@")[0]
                smoke_test_model(
                    Path(models_dir) / simplex_model, ["--modified-bases", modbase]
                )
