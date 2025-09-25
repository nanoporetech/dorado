#! /usr/bin/env python3

import pathlib
import shutil
import unittest

from tetra.compare_platforms import PlatformCompare, SpecialCase, compare_platforms


class TestComparePlatforms(unittest.TestCase):

    def setUp(self):
        self.data_dir = pathlib.Path(__file__).parent / "data"
        self.input_data = self.data_dir / "input"

    def tearDown(self):
        pass

    def remove_output_folder(self):
        shutil.rmtree(self.output_data)

    def test_within_tolerance(self):
        comparison = PlatformCompare(
            reference_path=self.input_data / "platform_compare_test_data",
            platform_1="platform1",
            platform_2="platform2",
            platform_tolerance=0.02,
            special_cases=[SpecialCase(pattern="barcoding_tests", tolerance=None)],
            exclude_columns=["passes_filtering"],
        )
        failures = compare_platforms(comparison)
        self.assertFalse(failures)

    def test_exceeds_tolerance(self):
        comparison = PlatformCompare(
            reference_path=self.input_data / "platform_compare_test_data",
            platform_1="platform1",
            platform_2="platform2",
            platform_tolerance=0.005,
            special_cases=[SpecialCase(pattern="barcoding_tests", tolerance=None)],
            exclude_columns=["passes_filtering"],
        )
        failures = compare_platforms(comparison)
        self.assertEqual(len(failures), 1)
        windows_safe_path = pathlib.Path("basecalling_tests") / "sequencing_summary.txt"
        self.assertIn(str(windows_safe_path), failures[0])
        self.assertIn("expected 1500.0", failures[0])
        self.assertIn("actual 1515.0", failures[0])
        self.assertIn("expected 12.5", failures[0])
        self.assertIn("actual 12.6", failures[0])


if __name__ == "__main__":
    unittest.main()
