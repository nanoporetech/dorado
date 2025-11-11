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

# The basic regression tests expect the `regression_test_data` project
# to have been checked out into the root folder of `dorado`. If it is in
# another location you can override the default with the `INPUT_FOLDER`
# environment variable. For benchmarking tests you will always need to
# use the override to point to the location the data has been downloaded
# to.
INPUT_FOLDER_OVERRIDE = os.getenv("INPUT_FOLDER")
if INPUT_FOLDER_OVERRIDE is None:
    INPUT_FOLDER = ROOT_DIR / "regression_test_data"
else:
    INPUT_FOLDER = pathlib.Path(INPUT_FOLDER_OVERRIDE)
