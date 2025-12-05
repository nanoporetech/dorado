import abc
import os
import typing
import unittest
from dataclasses import dataclass, field
from functools import cache

from data_paths import INPUT_FOLDER, OUTPUT_FOLDER, ROOT_DIR
from tetra.timestamped_context import TimestampedContext
from tetra.utilities import get_platform, run_with_timeout


def _get_benchmarking_script_args(
    preset: str,
    basecall_model: str,
    input_read: str,
    output_file: str,
    duration: int | None,
):
    gpu_devices = {
        "macos": "metal",
        "orin": "cuda:0",
        "gridion": "cuda:0",
        "p24": "cuda:0,1,2,3",
    }

    # use the py launcher on Windows to utilise the shebang notation in the script
    args = ["py"] if get_platform() == "windows" else []

    args += [
        _get_benchmark_script_location(),
        "--device",
        gpu_devices[preset],
        "--basecall_model",
        basecall_model,
        "--input_file",
        input_read,
        "--output_file",
        output_file,
    ]

    if duration is not None:
        args += ["--run_for", str(duration)]

    # Set the cache_directory from the CACHE_DIR env var
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir is not None:
        args.extend(["--models_directory", cache_dir])

    return args


def _get_benchmark_script_location():
    return str(ROOT_DIR / "regression_test" / "benchmarking.py")


@dataclass
class kit14:
    fast_model: str = "fast"
    hac_model: str = "hac"
    sup_model: str = "sup"

    mod_5mC_5hmC: str = "5mC_5hmC"
    mod_5mCG_5hmCG: str = "5mCG_5hmCG"
    mod_6mA: str = "6mA"

    suffix: str = "_kit14"

    test_conditions: typing.Dict[str, typing.List[str]] = field(init=False)
    test_conditions_modbase: typing.Dict[str, typing.List[str]] = field(init=False)
    test_conditions_modbase_v100_prom: typing.Dict[str, typing.List[str]] = field(
        init=False
    )

    def __post_init__(self):
        # modelsets should be the model complex sections for the basecalling model, comma-separated mods and "duplex"
        # e.g. ["fast@v5.0.0", "5mC,6mA", "duplex"]
        model_sets = [
            [self.fast_model, None],
            [self.hac_model, None],
            [self.sup_model, None],
        ]

        self.test_conditions = {
            "gridion": model_sets,
            "p24": model_sets,
            "orin": model_sets,
            "macos": model_sets,
        }

        modbase_model_sets = [
            [self.hac_model, self.mod_5mCG_5hmCG],
            [self.hac_model, self.mod_5mC_5hmC],
            [self.hac_model, self.mod_6mA],
            [
                self.hac_model,
                ",".join(
                    [
                        self.mod_5mCG_5hmCG,
                        self.mod_6mA,
                    ]
                ),
            ],
            [
                self.hac_model,
                ",".join(
                    [
                        self.mod_5mC_5hmC,
                        self.mod_6mA,
                    ]
                ),
            ],
            [self.sup_model, self.mod_5mCG_5hmCG],
            [self.sup_model, self.mod_5mC_5hmC],
            [self.sup_model, self.mod_6mA],
            [
                self.sup_model,
                ",".join(
                    [
                        self.mod_5mCG_5hmCG,
                        self.mod_6mA,
                    ]
                ),
            ],
            [
                self.sup_model,
                ",".join(
                    [
                        self.mod_5mC_5hmC,
                        self.mod_6mA,
                    ]
                ),
            ],
        ]

        self.test_conditions_modbase = {
            "gridion": modbase_model_sets,
            "p24": modbase_model_sets,
            "orin": modbase_model_sets,
            "macos": modbase_model_sets,
        }

        # restricted set of modbase tests specifically for the V100 prom,
        # because we're running too many tests on a platform we're not that concerned about
        modbase_model_sets_v100_prom = [
            [self.hac_model, self.mod_5mCG_5hmCG],
            [self.hac_model, self.mod_5mC_5hmC],
            [
                self.hac_model,
                ",".join(
                    [
                        self.mod_5mC_5hmC,
                        self.mod_6mA,
                    ]
                ),
            ],
            [self.sup_model, self.mod_5mCG_5hmCG],
            [self.sup_model, self.mod_5mC_5hmC],
            [
                self.sup_model,
                ",".join(
                    [
                        self.mod_5mC_5hmC,
                        self.mod_6mA,
                    ]
                ),
            ],
        ]

        self.test_conditions_modbase_v100_prom = {"p24": modbase_model_sets_v100_prom}


class BasecallingSpeedTestCases(abc.ABC):
    """Base class containing the tests to be run. Inherit from this and override the various test conditions
    to set up benchmarking runs for different sets of configurations.
    """

    @property
    @abc.abstractmethod
    def kit(self):
        """Kit object"""
        pass

    @property
    @abc.abstractmethod
    def standard_reads(self):
        """Filename of benchmarking data"""
        pass

    @property
    @abc.abstractmethod
    def suffix(self):
        """Folder suffix for result output files"""
        pass

    @property
    def device(self):
        """Device name to add to the output file name"""
        return os.getenv("GPU_DEVICE_NAME")

    # All test conditions are dicts of device identifier to array of config names.
    @property
    def test_conditions(self):
        # Filter by platform and/or device name
        if get_platform() == "orin":
            return {"orin": self.kit.test_conditions["orin"]}
        elif get_platform() == "osx_arm":
            return {"macos": self.kit.test_conditions["macos"]}
        elif self.device == "A6000":
            return {"gridion": self.kit.test_conditions["gridion"]}
        elif self.device == "A100":
            return {"p24": self.kit.test_conditions["p24"]}
        else:
            return {
                "gridion": self.kit.test_conditions["gridion"],
                # "p24": self.kit.test_conditions["p24"],
            }

    @property
    def test_conditions_modbase(self):
        if get_platform() == "orin":
            return {"orin": self.kit.test_conditions_modbase["orin"]}
        elif get_platform() == "osx_arm":
            return {"macos": self.kit.test_conditions_modbase["macos"]}
        elif self.device == "A6000":
            return {"gridion": self.kit.test_conditions_modbase["gridion"]}
        elif self.device == "A100":
            return {"p24": self.kit.test_conditions_modbase["p24"]}
        elif self.device == "V100":
            return {
                "gridion": self.kit.test_conditions_modbase["gridion"],
                "p24": self.kit.test_conditions_modbase_v100_prom["p24"],
            }
        else:
            return {
                "gridion": self.kit.test_conditions_modbase["gridion"],
                "p24": self.kit.test_conditions_modbase["p24"],
            }

    def run_benchmark(
        self,
        input_read_filename: str,
        test_folder: str,
        test_conditions: dict,
        reference: str | None = None,
        duration: int | None = None,
        extended_args: list | None = None,
    ):
        """Runs a set of speed benchmarks.

        :param input_read_filename: the base filename of a pod5 file to use when
            benchmarking. This file should be in a folder called "regression_test_data"
            in the root directory of the project. The location can be overridden with
            the `INPUT_FOLDER` environment variable.
        :param test_folder: The name of a test folder to store these benchmark
            results in, e.g. "benchmarking_standard". This folder will be
            automatically created in the appropriate regression test location by
            tetra. Must be unique relative to other benchmarking test folder
            names.
        :param test_conditions: A dictionary mapping presets to a list of
            modelsets to evaluate for that preset (see examples above).
        :param reference: Optional alignment reference. This file should be in the same
            folder as the input read file.
        :param duration: Optional duration to override the default benchmarking duration.
        :param extended_args: Additional optional arguments that should be passed to the
            benchmarking script.
        """
        device_suffix = "" if self.device is None else "_" + self.device
        test_folder = test_folder + device_suffix + self.suffix

        input_read = str(INPUT_FOLDER / input_read_filename)
        if extended_args is None:
            extended_args = []
        if reference is not None:
            ref_file = str(INPUT_FOLDER / reference)
            extended_args.extend(["--reference", ref_file])

        for preset in test_conditions:
            for modelset in test_conditions[preset]:
                assert len(modelset) == 2
                # output name - concatenate the model and mods and replace any commas (e.g. if there are multiple mods)
                output_name = "-".join([x for x in modelset if x is not None])
                output_name = output_name.replace(",", "-")
                output_filename = "{}_{}.txt".format(preset, output_name)
                output_file = os.path.join(OUTPUT_FOLDER, test_folder, output_filename)
                script_args = _get_benchmarking_script_args(
                    preset,
                    modelset[0],
                    input_read,
                    output_file,
                    duration,
                )
                script_args += extended_args
                if modelset[1] is not None:
                    script_args += ["--modified_bases", modelset[1]]

                with self.subTest(preset=preset, modelset=modelset):
                    with TimestampedContext(
                        f"BasecallingSpeedTestCases: {preset} {modelset}"
                    ):
                        try:
                            run_with_timeout(script_args, None)
                        except AssertionError as msg:
                            print(f"Regression test {output_filename} failed: {msg}")

    def test_standard_basecalling_speeds(self):
        self.run_benchmark(
            self.standard_reads, "bm_basecalling_standard", self.test_conditions
        )

    def test_precise_condition_speeds(self):
        model_sets = [
            [
                self.kit.hac_model,
                self.kit.mod_5mCG_5hmCG,
            ],
        ]
        test_conditions = dict.fromkeys(self.test_conditions.keys(), model_sets)
        extra_args = ["--kit_name", "SQK-NBD114-96"]

        self.run_benchmark(
            self.standard_reads,
            "bm_basecall_precise",
            test_conditions,
            reference="humanGRCh38.mmi",
            extended_args=extra_args,
        )

    # Testing all the remora models takes a while, so we split them up to let
    # other jobs get into the queue -- we'll invoke each of these jobs
    # separately in CI.
    def run_modbase_speeds_for_model(self, model: str):
        test_conditions = {
            preset: [modelset for modelset in modelsets if model in modelset[0]]
            for (preset, modelsets) in self.test_conditions_modbase.items()
        }
        self.run_benchmark(
            self.standard_reads,
            "bm_modbase",
            test_conditions,
            duration=240,
        )

    def test_modbase_basecalling_speeds_hac(self):
        self.run_modbase_speeds_for_model(self.kit.hac_model)

    def test_modbase_basecalling_speeds_sup(self):
        self.run_modbase_speeds_for_model(self.kit.sup_model)


class TestBasecallingSpeeds_kit14_30kbp(BasecallingSpeedTestCases, unittest.TestCase):
    @property
    @cache
    def kit(self):
        return kit14()

    @property
    def standard_reads(self):
        return "r10.4.1_5khz_30kbp_unsorted.pod5"

    @property
    def suffix(self):
        return self.kit.suffix + "_30kbp"


class TestBasecallingSpeeds_kit14_200bp(BasecallingSpeedTestCases, unittest.TestCase):
    @property
    def kit(self):
        return kit14()

    @property
    def standard_reads(self):
        return "r10.4.1_5khz_200bp.pod5"

    @property
    def suffix(self):
        return self.kit.suffix + "_200bp"
