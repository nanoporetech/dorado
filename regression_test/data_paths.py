import os
import pathlib

from tetra.utilities import get_platform

GPU_BUILD = os.getenv("BUILD_CUDA", "off").lower() in ("true", "1", "on")
PLATFORM = get_platform()
ROOT_DIR = pathlib.Path(__file__).parent.parent
_input_folder_override = os.getenv("INPUT_FOLDER")
if _input_folder_override is not None:
    INPUT_FOLDER = pathlib.Path(_input_folder_override)
else:
    INPUT_FOLDER = ROOT_DIR / "regression_test_data"
OUTPUT_FOLDER = ROOT_DIR / "regression_test" / "output" / PLATFORM
REFERENCE_FOLDER = ROOT_DIR / "regression_test" / "ref" / PLATFORM
