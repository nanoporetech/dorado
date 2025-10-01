import os
import pathlib

from tetra.utilities import get_platform

GPU_BUILD = bool(os.environ.get("BUILD_CUDA"))
PLATFORM = get_platform()
ROOT_DIR = pathlib.Path(__file__).parent.parent
input_folder = ROOT_DIR / "regression_test_data"
output_folder = ROOT_DIR / "regression_test" / "output" / PLATFORM
reference_folder = ROOT_DIR / "regression_test" / "ref" / PLATFORM
