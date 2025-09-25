#! /usr/bin/env python3

import pathlib
import shutil
import unittest

from tetra.regression_manager import RegressionManager, TestFile, TestResult


class TestRegressionManager(unittest.TestCase):

    KEEP_OUTPUT_DATA = False  # Set to True to keep file for debugging

    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"
        self.input_data = self.data_dir / "input"
        self.output_data = self.data_dir / "output"
        if self.output_data.exists():
            self.remove_output_folder()
        pathlib.Path.mkdir(
            self.output_data, exist_ok=TestRegressionManager.KEEP_OUTPUT_DATA
        )

    def tearDown(self):
        if not TestRegressionManager.KEEP_OUTPUT_DATA:
            self.remove_output_folder()

    def remove_output_folder(self):
        shutil.rmtree(self.output_data)

    def make_test_data(self):
        test_files = [
            TestFile(
                name="result1.txt",
                subfolder="condition1",
                exists_in_ref=True,
                exists_in_output=True,
                matches=True,
                validation_passed=False,
            ),
            TestFile(
                name="result1.fastq",
                subfolder="condition1",
                exists_in_ref=True,
                exists_in_output=True,
                matches=False,
                validation_passed=False,
            ),
            TestFile(
                name="result2.txt",
                subfolder="condition2",
                exists_in_ref=True,
                exists_in_output=False,
                matches=None,
                validation_passed=None,
            ),
            TestFile(
                name="result2.fastq",
                subfolder="condition2",
                exists_in_ref=True,
                exists_in_output=False,
                matches=None,
                validation_passed=None,
            ),
        ]
        return test_files

    def test_basic_usage(self):
        manager = RegressionManager()
        manager.open_test("test", "test_dir")

        test_files = self.make_test_data()
        manager.add_test_files(test_files)
        manager.close_test(TestResult.COMPLETED)

        data = manager.results
        self.assertEqual(len(data.tests), 1)
        test = data.tests[0]
        self.assertEqual(test.name, "test")
        self.assertEqual(test.test_folder, "test_dir")
        self.assertEqual(test.subfolders, set(["condition1", "condition2"]))
        self.assertEqual(test.test_files, test_files)
        self.assertEqual(test.result, TestResult.COMPLETED)

    def test_collect_files(self):
        manager = RegressionManager()
        manager.open_test("test", "test_dir")
        manager.set_main_output_folder(self.input_data / "collect_files_test_data")

        all_test_files = manager.collect_files(None, {"txt": 5, "fastq": 3})
        self.assertEqual(len(all_test_files), 8)

        test_files1 = manager.collect_files("subdir1", {"txt": 3, "fastq": 1})
        self.assertEqual(len(test_files1), 4)

        test_files2 = manager.collect_files("subdir2", {"txt": 1, "fastq": 2})
        self.assertEqual(len(test_files2), 3)

        # Gotta do this to prevent windows backslash issues.
        subdir1_1 = str(pathlib.Path("subdir1") / "subdir1_1")
        expected = {
            "test_file1.txt": None,
            "test_file2.txt": "subdir1",
            "test_file3.fastq": "subdir1",
            "test_file4.txt": subdir1_1,
            "test_file5.txt": subdir1_1,
            "test_file6.txt": "subdir2",
            "test_file7.fastq": "subdir2",
            "test_file8.fastq": "subdir2",
        }
        for test_file in all_test_files:
            self.assertIn(test_file.name, expected)
            self.assertEqual(test_file.subfolder, expected[test_file.name])

        manager.close_test(TestResult.ABORTED)

    def test_serialization(self):
        manager = RegressionManager()
        manager.open_test("test", "test_dir")

        test_files = self.make_test_data()
        manager.add_test_files(test_files)
        manager.close_test(TestResult.COMPLETED)

        json_file = self.output_data / "test_data.json"
        manager.write_to_file(json_file)
        self.assertTrue(json_file.exists())

        loaded_mgr = RegressionManager.read_from_file(json_file)

        data = loaded_mgr.results
        self.assertEqual(len(data.tests), 1)
        test = data.tests[0]
        self.assertEqual(test.name, "test")
        self.assertEqual(test.test_folder, "test_dir")
        self.assertEqual(test.subfolders, set(["condition1", "condition2"]))
        self.assertEqual(test.test_files, test_files)
        self.assertEqual(test.result, TestResult.COMPLETED)


if __name__ == "__main__":
    unittest.main()
