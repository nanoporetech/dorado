import os
import pathlib
import contextlib
import subprocess
import shutil
import typing
import unittest
from copy import deepcopy

from data_paths import (
    PLATFORM,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    REFERENCE_FOLDER,
)

from tetra.data_checker import DataChecker
from tetra.regression_context import RegressionContext
from tetra.regression_manager import RegressionManager, TestData, TestResult
from tetra.sequence_utils import USE_PYSAM
from tetra.update_reference_data import run_update

DEBUG = os.getenv("BUILD_TYPE", "Release").upper() == "DEBUG"
DEFAULT_MAX_TIMEOUT = 300
VALIDATION_OPTIONS = {"gpu_calling_enabled": True}


class TestDorado(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This initializes the RegressionManager object, so that it can persist
        through all of the tests. We create the output folder if it doesn't
        exist, even though the test executables should do so, just in case
        something goes very wrong and none of the executables create the path.
        That way the json file can still be written.
        """
        cls.manager = RegressionManager()
        cls.context = RegressionContext(cls.manager, OUTPUT_FOLDER)
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        OUTPUT_FOLDER.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        """
        Once all tests are complete, we need to finalize the RegressionManager.
        We then run the reference-update code, but only to get a report on what
        updates would be done by a rebase, without modifying the reference files.
        """
        if cls.manager.is_test_open:
            cls.manager.close_test(TestResult.ABORTED)
        json_summary_file = OUTPUT_FOLDER / "regression_test_results.json"
        cls.manager.write_to_file(json_summary_file)
        run_update(
            test_folder=OUTPUT_FOLDER.parent.parent,
            platform=PLATFORM,
            update_refs=False,
            quiet_mode=True,
        )

    def test_basecalling(self):
        """
        Test basic basecalling functionality dorado.
        """
        runs = [
            {
                "folder": "Kit14_fast",
                "input": "SQK-LSK114",
                "model": "fast",
            },
            {
                "folder": "Kit14_hac",
                "input": "SQK-LSK114",
                "model": "hac",
            },
            {
                "folder": "Kit14_sup",
                "input": "SQK-LSK114",
                "model": "sup",
            },
            {
                "folder": "Kit14_RNA_fast",
                "input": "SQK-RNA004",
                "model": "fast",
            },
            {
                "folder": "Kit14_RNA_hac",
                "input": "SQK-RNA004",
                "model": "hac",
            },
            {
                "folder": "Kit14_RNA_sup",
                "input": "SQK-RNA004",
                "model": "sup",
            },
        ]

        test_name = "basecalling"
        with self.context.open_test(self, test_name) as context:
            # All of these tests should produce a single .bam file and a single .txt summary file.
            if USE_PYSAM:
                expected_files = {"bam": 1, "txt": 1}
            else:
                expected_files = {"txt": 1}
            validation_settings = deepcopy(VALIDATION_OPTIONS)

            for run in runs:
                with self.context.open_subtest("test_basecalling", line=run):
                    subfolder = run["folder"]
                    output_folder = OUTPUT_FOLDER / test_name / subfolder
                    output_file = pathlib.Path("out.bam")
                    dorado_args = self.get_dorado_args(
                        input_path=INPUT_FOLDER / run["input"],
                        save_path=None,
                        model=run["model"],
                        emit_fastq=False,
                        emit_summary=True,
                    )

                    # Temporary fix for DOR-1428. Fix SUP batchsize for orin to 32 to
                    # prevent non-deterministic results.
                    _orin_sup_batchsize_args(run["model"], dorado_args)

                    errors = None
                    try:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        with contextlib.chdir(output_folder):
                            with output_file.open("wb") as outfile:
                                run_dorado(
                                    dorado_args, DEFAULT_MAX_TIMEOUT, outfile=outfile
                                )
                            # Now generate a post-run summary file, which we check for regressions but don't validate against the spec.
                            # Specifically make this a ".tsv" file so we don't mix it up with the inline summary
                            make_summary(
                                output_file, "summary.tsv", DEFAULT_MAX_TIMEOUT
                            )
                    except Exception as ex:
                        msg = f"Error checking output files for 'test_basecalling {subfolder}'.\n{ex}"
                        context.encountered_error()
                        self.fail(msg)

                    dorado_errors = self.check_program_output(
                        test_name,
                        subfolder,
                        expected_files,
                        validation_settings,
                    )
                    summary_errors = self.check_program_output(
                        test_name, subfolder, {"tsv": 1}, None
                    )

                    errors = "\n".join(filter(None, (dorado_errors, summary_errors)))
                    if errors is not None:
                        # This indicates regression test failures due to file comparison and/or validation,
                        # but not due to an exception being thrown or the executable crashing.
                        self.fail(errors)

    def test_modbase(self):
        """
        Test basic basecalling with base modifications.

        See Jira ticket DOR-1405 for details on validation process before committing updated
        regression test data.
        """
        runs = [
            {
                "folder": "HAC_4mC_5mC_6mA",
                "input": "modbase_DNA",
                "model": "hac,4mC_5mC,6mA",
            },
            {
                "folder": "HAC_5mC_5hmC",
                "input": "modbase_DNA",
                "model": "hac,5mC_5hmC",
            },
            {
                "folder": "HAC_5mCG_5hmCG",
                "input": "modbase_DNA",
                "model": "hac,5mCG_5hmCG",
            },
            {
                "folder": "SUP_4mC_5mC_6mA",
                "input": "modbase_DNA",
                "model": "sup,4mC_5mC,6mA",
            },
            {
                "folder": "SUP_5mC_5hmC",
                "input": "modbase_DNA",
                "model": "sup,5mC_5hmC",
            },
            {
                "folder": "SUP_5mCG_5hmCG",
                "input": "modbase_DNA",
                "model": "sup,5mCG_5hmCG",
            },
            {
                "folder": "HAC_inosine_m6A_m5C",
                "input": "modbase_RNA",
                "model": "hac,inosine_m6A,m5C",
            },
            {
                "folder": "HAC_m6A_DRACH_pseU",
                "input": "modbase_RNA",
                "model": "hac,m6A_DRACH,pseU",
            },
            {
                "folder": "SUP_m6A_DRACH_pseU_2OmeU",
                "input": "modbase_RNA",
                "model": "sup,m6A_DRACH,pseU_2OmeU",
            },
            {
                "folder": "SUP_inosine_m6A_2OmeA_m5C_2OmeC_2OmeG",
                "input": "modbase_RNA",
                "model": "sup,inosine_m6A_2OmeA,m5C_2OmeC,2OmeG",
            },
        ]

        test_name = "modified_basecalling"
        with self.context.open_test(self, test_name) as context:
            # All of these tests should produce a single .bam file and a single .txt summary file.
            if USE_PYSAM:
                expected_files = {"bam": 1, "txt": 1}
            else:
                expected_files = {"txt": 1}
            validation_settings = deepcopy(VALIDATION_OPTIONS)
            validation_settings["modified_bases_enabled"] = True

            for run in runs:
                with self.context.open_subtest("test_modified_basecalling", line=run):
                    subfolder = run["folder"]
                    output_folder = OUTPUT_FOLDER / test_name / subfolder
                    output_file = pathlib.Path("out.bam")
                    dorado_args = self.get_dorado_args(
                        input_path=INPUT_FOLDER / run["input"],
                        save_path=None,
                        model=run["model"],
                        emit_fastq=False,
                        emit_summary=True,
                        recursive=True,
                    )

                    # Temporary fix for DOR-1428. Fix SUP batchsize for orin to 32 to
                    # prevent non-deterministic results.
                    _orin_sup_batchsize_args(run["model"], dorado_args)

                    errors = None
                    try:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        with contextlib.chdir(output_folder):
                            with output_file.open("wb") as outfile:
                                run_dorado(
                                    dorado_args, DEFAULT_MAX_TIMEOUT, outfile=outfile
                                )
                            make_summary(
                                output_file, "summary.tsv", DEFAULT_MAX_TIMEOUT
                            )
                    except Exception as ex:
                        msg = f"Error checking output files for 'test_basecalling {subfolder}'.\n{ex}"
                        context.encountered_error()
                        self.fail(msg)

                    dorado_errors = self.check_program_output(
                        test_name, subfolder, expected_files, validation_settings
                    )

                    summary_errors = self.check_program_output(
                        test_name, subfolder, {"tsv": 1}, None
                    )

                    errors = "\n".join(filter(None, (dorado_errors, summary_errors)))
                    if errors is not None:
                        # This indicates regression test failures due to file comparison and/or validation,
                        # but not due to an exception being thrown or the executable crashing.
                        self.fail(errors)

    def check_program_output(
        self,
        test_name: str,
        subfolder: str,
        expected_files: dict,
        validation_settings: dict | None,
    ) -> str | None:
        """
        This functionality should be common to most, if not all, tests. It will gather the
        expected output files that need to be checked, see if they are present in the reference
        data, and if they are, perform the comparison. It then updates the RegressionManager
        with the results, and returns a report in string form of any mismatch between reference
        and output files.
        """
        test_files = self.manager.collect_files(subfolder, expected_files)
        test_data = TestData(
            name=test_name,
            test_folder=test_name,
            subfolders=None,
            test_files=test_files,
            result=None,
        )
        checker = DataChecker(str(REFERENCE_FOLDER), str(OUTPUT_FOLDER))
        results = checker.check_test_data(test_data, validation_settings)
        self.manager.add_test_files(test_files)
        errors = DataChecker.make_error_message(results)
        return errors

    def get_dorado_args(
        self,
        input_path: pathlib.Path,
        save_path: pathlib.Path | None,
        model: str,
        emit_fastq: bool,
        emit_summary: bool,
        recursive: bool = False,
    ) -> list:
        device = "metal" if PLATFORM == "osx_arm" else "cuda:0"
        args = ["doradod"] if DEBUG else ["dorado"]
        args.extend(
            [
                "basecaller",
                model,
                str(input_path),
                "--device",
                device,
            ]
        )
        if save_path is not None:
            args.extend(
                [
                    "--output-dir",
                    str(save_path),
                ]
            )
        if emit_fastq:
            args.append("--emit-fastq")
        if emit_summary:
            args.append("--emit-summary")
        if recursive:
            args.append("--recursive")
        return args


def run_dorado(cmd_args: list, timeout: int, outfile: typing.IO | None = None):
    print("Dorado command line: ", " ".join(cmd_args))
    out = subprocess.PIPE if outfile is None else outfile
    result = subprocess.run(
        cmd_args, timeout=timeout, text=True, stdout=out, stderr=subprocess.PIPE
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


def _orin_sup_batchsize_args(model: str, dorado_args: typing.List[str]):
    if PLATFORM == "orin" and model.startswith("sup"):
        dorado_args.extend(["--batchsize", "32"])


if __name__ == "__main__":
    unittest.main()
