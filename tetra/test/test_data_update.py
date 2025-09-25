#! /usr/bin/env python3

import pathlib
import shutil
import subprocess
import unittest
from datetime import datetime

from tetra.sequence_utils import USE_PYSAM


def run_with_timeout(cmd_args: list[str], timeout: int, verbose: bool):
    try:
        if verbose:
            print("Command line:", " ".join(cmd_args))
        start_time = datetime.now()
        subprocess.check_call(cmd_args, timeout=timeout, text=True)
        return (datetime.now() - start_time).seconds
    except subprocess.CalledProcessError as e:
        print("Error running command line:", " ".join(cmd_args))
        print("return code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise


@unittest.skipUnless(USE_PYSAM, "Pysam unavailable")
class TestDataUpdate(unittest.TestCase):

    KEEP_OUTPUT_DATA = False  # Set to True to keep files for debugging
    TIMEOUT = 60

    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"
        self.input_data = self.data_dir / "input"
        self.output_data = self.data_dir / "output"
        if self.output_data.exists():
            self.remove_output_folder()
        pathlib.Path.mkdir(self.output_data, exist_ok=TestDataUpdate.KEEP_OUTPUT_DATA)

    def tearDown(self):
        if not TestDataUpdate.KEEP_OUTPUT_DATA:
            self.remove_output_folder()

    def remove_output_folder(self):
        shutil.rmtree(self.output_data)

    def setup_test_data(
        self, test_folder: str, platforms: list[str], json_summary: pathlib.Path
    ) -> tuple[pathlib.Path, pathlib.Path]:
        test_path = self.output_data / test_folder
        pathlib.Path.mkdir(test_path, exist_ok=False)
        ref = test_path / "ref"
        pathlib.Path.mkdir(ref, exist_ok=False)
        out = test_path / "output"
        pathlib.Path.mkdir(out, exist_ok=False)
        rebase_data = self.input_data / "rebase_test_data"
        for platform in platforms:
            platform_ref = ref / platform
            shutil.copytree(rebase_data, platform_ref)
            platform_out = out / platform
            shutil.copytree(rebase_data, platform_out)
            platform_summary = platform_out / "regression_test_results.json"
            shutil.copyfile(json_summary, platform_summary)
        return ref, out

    def run_update_script(
        self, test_folder: str, update_refs: bool = True
    ) -> str | None:
        test_path = self.output_data / test_folder
        script_folder = pathlib.Path(__file__).parent.parent / "tetra"
        script = script_folder / "update_reference_data.py"
        args = [
            "python3",
            str(script),
            str(test_path),
            "--quiet",
        ]
        if update_refs:
            args.append("--update_refs")
        run_with_timeout(args, TestDataUpdate.TIMEOUT, verbose=False)
        summary = test_path / "output" / "regression_summary.txt"
        if not summary.exists():
            return None
        with summary.open("r") as file:
            file_content = file.read()
        return file_content

    def check_in_summary(self, summary: str, query: str):
        self.assertIn(
            query, summary, f"Failed to find string '{query}' in rebase summary."
        )

    def check_summary_line_count(self, summary: str, count: int):
        lines = summary.splitlines()
        self.assertEqual(
            len(lines),
            count,
            f"{len(lines)} lines found in rebase summary, {count} lines expected.",
        )

    def verify_rebase(self, test_folder: str, platforms: list[str]):
        test_path = self.output_data / test_folder
        out = test_path / "output"
        json_summary = self.input_data / "test_data.json"
        for platform in platforms:
            platform_out = out / platform
            platform_summary = platform_out / "regression_test_results.json"
            shutil.copyfile(json_summary, platform_summary)
        summary = self.run_update_script(test_folder, update_refs=False)
        self.check_summary_line_count(summary, 1)
        self.check_in_summary(
            summary, "All regression tests passed. Nothing to rebase."
        )

    def test_all_tests_passed(self):
        # No rebase should happen.
        test_folder_name = "all_tests_passed"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 1)
        self.check_in_summary(
            summary, "All regression tests passed. Nothing to rebase."
        )

    def test_missing_json_summary(self):
        # No rebase should happen.
        test_folder_name = "missing_json_summary"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        json_file_to_remove = (
            self.output_data
            / test_folder_name
            / "output"
            / "linux"
            / "regression_test_results.json"
        )
        json_file_to_remove.unlink()
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 3)
        self.check_in_summary(
            summary, "Cannot rebase reference files, due to the following errors:"
        )
        self.check_in_summary(summary, "Platform folder: linux")
        self.check_in_summary(summary, "ERROR: Test results file")

    def test_bad_json_summary(self):
        # No rebase should happen.
        test_folder_name = "bad_json_summary"
        json_summary = self.input_data / "test_data_bad.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 5)
        self.check_in_summary(
            summary, "Cannot rebase reference files, due to the following errors:"
        )
        self.check_in_summary(summary, "Platform folder: linux")
        self.check_in_summary(summary, "Platform folder: windows")
        self.check_in_summary(summary, "ERROR: Failed to parse file")

    def test_aborted_regression_tests(self):
        # No rebase should happen.
        test_folder_name = "aborted_regression_test"
        json_summary = self.input_data / "test_data_aborted.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 5)
        self.check_in_summary(
            summary, "Cannot rebase reference files, due to the following errors:"
        )
        self.check_in_summary(summary, "Platform folder: linux")
        self.check_in_summary(summary, "Platform folder: windows")
        self.check_in_summary(
            summary,
            "ERROR: Test(s) were aborted or failed with errors that prevent rebasing.",
        )

    def test_extra_ref_platform(self):
        # No rebase should happen.
        test_folder_name = "extra_ref_platform"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        extra_ref_platform_folder = (
            self.output_data
            / test_folder_name
            / "ref"
            / "osx_arm"
            / "basecalling"
            / "Kit14_cDNA_fast"
        )
        extra_ref_platform_folder.mkdir(parents=True)
        extra_file = extra_ref_platform_folder / "sequencing_summary.txt"
        extra_file.touch()
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 3)
        self.check_in_summary(
            summary, "Cannot rebase reference files, due to the following errors:"
        )
        self.check_in_summary(summary, "Platform folder: osx_arm")
        self.check_in_summary(summary, "ERROR: Output folder for platform is missing.")

    def test_missing_ref_platform(self):
        # Rebase should create missing platform folder and files.
        test_folder_name = "missing_ref_platform"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        folder_to_remove = self.output_data / test_folder_name / "ref" / "windows"
        shutil.rmtree(folder_to_remove)
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 23)
        self.check_in_summary(summary, "Rebasing can be performed.")
        self.check_in_summary(summary, "Platform folder: windows")
        self.check_in_summary(summary, "Files to be added to reference folder:")
        self.verify_rebase(test_folder_name, ["linux", "windows"])

    def test_missing_ref_folder(self):
        # Rebase should create missing folder and files.
        test_folder_name = "missing_ref_folder"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        folder_to_remove = (
            self.output_data
            / test_folder_name
            / "ref"
            / "windows"
            / "alignment_tests"
            / "RNA_mod_truth_sets_fast"
        )
        shutil.rmtree(folder_to_remove)
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 5)
        self.check_in_summary(summary, "Rebasing can be performed.")
        self.check_in_summary(summary, "Platform folder: windows")
        self.check_in_summary(summary, "Files to be added to reference folder:")
        self.verify_rebase(test_folder_name, ["linux", "windows"])

    def test_extra_ref_folder(self):
        # Rebase should remove all files in extra ref folder.
        test_folder_name = "extra_ref_folder"
        json_summary = self.input_data / "test_data.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)
        extra_ref_folder = (
            self.output_data
            / test_folder_name
            / "ref"
            / "linux"
            / "duplex"
            / "Kit14_cDNA_fast"
        )
        extra_ref_folder.mkdir(parents=True)
        extra_file = extra_ref_folder / "sequencing_summary.txt"
        extra_file.touch()
        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 4)
        self.check_in_summary(summary, "Rebasing can be performed.")
        self.check_in_summary(summary, "Platform folder: linux")
        self.check_in_summary(summary, "Files to be removed from reference folder:")
        self.verify_rebase(test_folder_name, ["linux", "windows"])

    def test_standard_rebase(self):
        # Rebase should update all files that mismatches, add any
        # missing files, and remove any excess files.
        test_folder_name = "standard_rebase"
        json_summary = self.input_data / "test_data_standard_rebase.json"
        self.setup_test_data(test_folder_name, ["linux", "windows"], json_summary)

        # Add an extra folder to ref that will need to be removed.
        extra_ref_folder = (
            self.output_data
            / test_folder_name
            / "ref"
            / "linux"
            / "duplex"
            / "Kit14_cDNA_fast"
        )
        extra_ref_folder.mkdir(parents=True)
        extra_file = extra_ref_folder / "sequencing_summary.txt"
        extra_file.touch()

        # Remove a folder from ref, so that it will have to be copied over.
        for platform in ["linux", "windows"]:
            folder_to_remove = (
                self.output_data
                / test_folder_name
                / "ref"
                / platform
                / "alignment_tests"
                / "RNA_mod_truth_sets_fast"
            )
            shutil.rmtree(folder_to_remove)

        # Modify some files so that they will have to be replaced in ref.
        for platform in ["linux", "windows"]:
            changed_file = (
                self.input_data
                / "substitute_files"
                / "fastq_runid_9bf51088-8205-4cfa-9c95-98867d24702e_0_0.fastq"
            )
            path_in_output = (
                self.output_data
                / test_folder_name
                / "output"
                / platform
                / "barcoding_tests"
                / "FLO-MIN114_SQK-NBD114-96_fast"
                / "barcode22"
            )
            output_file = (
                path_in_output
                / "fastq_runid_9bf51088-8205-4cfa-9c95-98867d24702e_0_0.fastq"
            )
            shutil.copyfile(changed_file, output_file)

        summary = self.run_update_script(test_folder_name)
        self.check_summary_line_count(summary, 15)
        self.check_in_summary(summary, "Rebasing can be performed.")
        self.check_in_summary(summary, "Platform folder: linux")
        self.check_in_summary(summary, "Platform folder: windows")
        self.check_in_summary(summary, "Files to be removed from reference folder:")
        self.check_in_summary(summary, "Files to be added to reference folder:")
        self.check_in_summary(summary, "Files to be replaced in reference folder:")
        self.verify_rebase(test_folder_name, ["linux", "windows"])


if __name__ == "__main__":
    unittest.main()
