#! /usr/bin/env python3

import pathlib
import shutil
import unittest

from tetra.data_checker import DataChecker
from tetra.regression_manager import TestData, TestFile
from tetra.sequence_utils import USE_PYSAM


class TestDataChecker(unittest.TestCase):

    KEEP_OUTPUT_DATA = False  # Set to True to keep files for debugging

    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"
        self.input_data = self.data_dir / "input"
        self.output_data = self.data_dir / "output"
        if self.output_data.exists():
            self.remove_output_folder()
        pathlib.Path.mkdir(self.output_data, exist_ok=TestDataChecker.KEEP_OUTPUT_DATA)

    def tearDown(self):
        if not TestDataChecker.KEEP_OUTPUT_DATA:
            self.remove_output_folder()

    def remove_output_folder(self):
        shutil.rmtree(self.output_data)

    def test_missing_folder(self):
        # Missing ref folder is OK.
        ref_folder = str(self.input_data / "unobtanium")
        output_folder = str(self.output_data)
        DataChecker(ref_folder, output_folder)

        # Missing output folder should raise exception.
        ref_folder = str(self.input_data)
        output_folder = str(self.output_data / "unobtanium")
        with self.assertRaises(Exception) as context:
            DataChecker(ref_folder, output_folder)
        self.assertTrue(
            f"Expected output folder `{output_folder}` does not exist."
            in str(context.exception)
        )

    def test_missing_ref_file(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        ref_file1 = ref_folder / "sequencing_data1.fastq"
        ref_file2 = ref_folder / "sequencing_summary_missing.txt"
        dummy_file = ref_folder / "sequencing_summary1.txt"
        out_file1 = output_folder / "sequencing_data1.fastq"
        out_file2 = output_folder / "sequencing_summary_missing.txt"
        pathlib.Path.mkdir(output_folder, exist_ok=True)
        shutil.copyfile(ref_file1, out_file1)
        shutil.copyfile(dummy_file, out_file2)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=[],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="sequencing_data1.fastq",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
                TestFile(
                    name="sequencing_summary_missing.txt",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
            ]
        )
        results = checker.check_test_data(test_data, None)
        self.assertEqual(len(results["validation"]), 0)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertEqual(
            results["comparison"][0], f"Reference file `{ref_file2}' does not exist."
        )
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertTrue(test_data.test_files[0].matches)
        self.assertFalse(test_data.test_files[1].exists_in_ref)
        self.assertTrue(test_data.test_files[1].exists_in_output)
        self.assertTrue(test_data.test_files[1].matches is None)

    def test_missing_output_file(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        ref_file1 = ref_folder / "sequencing_data1.fastq"
        out_file1 = output_folder / "sequencing_data1.fastq"
        out_file2 = output_folder / "sequencing_summary1.txt"
        pathlib.Path.mkdir(output_folder, exist_ok=True)
        shutil.copyfile(ref_file1, out_file1)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=[],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="sequencing_data1.fastq",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
                TestFile(
                    name="sequencing_summary1.txt",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
            ]
        )
        results = checker.check_test_data(test_data, None)
        self.assertEqual(len(results["validation"]), 0)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertEqual(
            results["comparison"][0], f"Output file `{out_file2}' does not exist."
        )
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertTrue(test_data.test_files[0].matches)
        self.assertTrue(test_data.test_files[1].exists_in_ref)
        self.assertFalse(test_data.test_files[1].exists_in_output)
        self.assertTrue(test_data.test_files[1].matches is None)

    def test_everything_ok(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        ref_subfolder = ref_folder / "sub"
        output_subfolder = output_folder / "sub"
        ref_file1 = ref_subfolder / "sequencing_data1.fastq"
        ref_file2 = ref_folder / "sequencing_summary1.txt"
        out_file1 = output_subfolder / "sequencing_data1.fastq"
        out_file2 = output_folder / "sequencing_summary1.txt"

        pathlib.Path.mkdir(output_folder, exist_ok=True)
        pathlib.Path.mkdir(output_subfolder, exist_ok=True)
        shutil.copyfile(ref_file1, out_file1)
        shutil.copyfile(ref_file2, out_file2)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=["sub"],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="sequencing_data1.fastq",
                    subfolder="sub",
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
                TestFile(
                    name="sequencing_summary1.txt",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                ),
            ]
        )

        validation_settings = {}

        results = checker.check_test_data(test_data, validation_settings)
        self.assertTrue(len(results["validation"]) < 2)
        self.assertEqual(len(results["comparison"]), 0)
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertTrue(test_data.test_files[0].matches)
        self.assertTrue(test_data.test_files[1].exists_in_ref)
        self.assertTrue(test_data.test_files[1].exists_in_output)
        self.assertTrue(test_data.test_files[1].matches)

        if len(results["validation"]) == 0:
            self.assertTrue(test_data.test_files[0].validation_passed)
            self.assertTrue(test_data.test_files[1].validation_passed)
        else:
            self.assertTrue(test_data.test_files[0].validation_passed is None)
            self.assertTrue(test_data.test_files[1].validation_passed is None)
            self.assertEqual(
                results["validation"][0],
                "Warning. Validation was requested, but is not enabled.",
            )

    def test_fastq_mismatch(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        dummy_file = ref_folder / "sequencing_data2.fastq"
        out_file1 = output_folder / "sequencing_data1.fastq"
        pathlib.Path.mkdir(output_folder, exist_ok=True)
        shutil.copyfile(dummy_file, out_file1)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=[],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="sequencing_data1.fastq",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                )
            ]
        )
        results = checker.check_test_data(test_data, None)
        self.assertEqual(len(results["validation"]), 0)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertFalse(test_data.test_files[0].matches)

    def test_summary_mismatch(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        dummy_file = ref_folder / "sequencing_summary2.txt"
        out_file1 = output_folder / "sequencing_summary1.txt"
        pathlib.Path.mkdir(output_folder, exist_ok=True)
        shutil.copyfile(dummy_file, out_file1)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=[],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="sequencing_summary1.txt",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                )
            ]
        )
        results = checker.check_test_data(test_data, None)
        self.assertEqual(len(results["validation"]), 0)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertFalse(test_data.test_files[0].matches)

    @unittest.skipUnless(USE_PYSAM, "Pysam unavailable")
    def test_bam_mismatch(self):
        ref_folder = self.input_data / "checker_test_data"
        output_folder = self.output_data / "checker_test_data"
        dummy_file = ref_folder / "alignment_data2.bam"
        out_file1 = output_folder / "alignment_data1.bam"
        pathlib.Path.mkdir(output_folder, exist_ok=True)
        shutil.copyfile(dummy_file, out_file1)
        checker = DataChecker(str(self.input_data), str(self.output_data))
        test_data = TestData(
            name="checker_test",
            test_folder="checker_test_data",
            subfolders=[],
            test_files=[],
            result=None,
        )
        test_data.test_files.extend(
            [
                TestFile(
                    name="alignment_data1.bam",
                    subfolder=None,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                )
            ]
        )
        results = checker.check_test_data(test_data, None)
        self.assertEqual(len(results["validation"]), 0)
        self.assertEqual(len(results["comparison"]), 1)
        self.assertTrue(test_data.test_files[0].exists_in_ref)
        self.assertTrue(test_data.test_files[0].exists_in_output)
        self.assertFalse(test_data.test_files[0].matches)


if __name__ == "__main__":
    unittest.main()
