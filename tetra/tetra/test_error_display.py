import unittest

from numpy import nan
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from tetra.error_display import (
    col_match,
    find_value_differences,
    pretty_print_diffs,
    row_match,
)


class TestErrorDisplay(unittest.TestCase):
    def setUp(self):
        self.headers = ["filename", "result", "score", "name", "surname"]
        self.example_data = [
            ["file1", 1, 7, "alice", "alison"],
            ["file4", 3, 4, "bob", "bobson"],
            ["file5", 6, 4, "claire", "eclair"],
            ["file2", 1, 2, "david", "davison"],
            ["file3", 9, 3, "ellie", "elephant"],
            ["file6", 3, 2, "fred", "right said"],
        ]
        self.exp = DataFrame(self.example_data, columns=self.headers).set_index(
            "filename"
        )

    def test_row_match(self):
        mod_input = self.example_data
        mod_input[1] = ["file8", 3, 4, "bob"]
        act = DataFrame(mod_input, columns=self.headers).set_index("filename")
        match, exp_only, act_only = row_match(self.exp, act)
        self.assertEqual(match, (set(["file1", "file5", "file2", "file3", "file6"])))
        self.assertEqual(exp_only, (set(["file4"])))
        self.assertEqual(act_only, (set(["file8"])))

    def test_column_match(self):
        mod_headers = self.headers
        mod_headers[2] = "new_score"
        act = DataFrame(self.example_data, columns=mod_headers).set_index("filename")
        match, exp_only, act_only = col_match(self.exp, act)
        self.assertEqual(match, (set(["result", "name", "surname"])))
        self.assertEqual(exp_only, (set(["score"])))
        self.assertEqual(act_only, (set(["new_score"])))

    def test_filter(self):
        # Test we return an empty data frame from identical inputs
        self.assertTrue(find_value_differences(self.exp, self.exp).empty)
        # self.assertEqual(find_value_differences(self.exp, self.exp), None)

        mod_input = self.example_data
        mod_input[1] = ["file4", 3, 10, "bob", "bobson"]
        mod_input[2] = ["file5", 6, 5, "claire", "nope"]
        mod_input[4] = ["file3", 9, 3, "eille_rc", "yep"]
        act = DataFrame(mod_input, columns=self.headers).set_index("filename")

        diff_res = [
            ["file4", 6, nan, nan],
            ["file5", 1, nan, "eclair -> nope"],
            ["file3", nan, "ellie -> eille_rc", "elephant -> yep"],
        ]
        expected_diffs = DataFrame(
            diff_res, columns=["filename", "score", "name", "surname"]
        ).set_index("filename")
        diffs = find_value_differences(self.exp, act)

        # using pandas.testing since this allows NaN == NaN
        assert_frame_equal(diffs, expected_diffs)

    def test_pretty_print_diffs(self):
        act = self.exp.copy(deep=True)
        act.drop("file6", inplace=True)
        diffs = pretty_print_diffs(self.exp, act)
        self.assertEqual(diffs, "SizeError(1 Rows only in expected: file6)")


if __name__ == "__main__":
    unittest.main()
