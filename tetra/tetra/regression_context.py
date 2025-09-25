import pathlib
import unittest
from contextlib import contextmanager

from tetra.regression_manager import RegressionManager, TestResult
from tetra.timestamped_context import TimestampedContext


class TestContext(object):
    def __init__(
        self, test_name: str, manager: RegressionManager, output_folder: pathlib.Path
    ):
        self._test_name = test_name
        self._manager = manager
        self._output_folder = output_folder
        self._no_errors = True

    def __enter__(self):
        self._manager.open_test(self._test_name, self._test_name)
        self._manager.set_main_output_folder(self._output_folder)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        status = TestResult.COMPLETED if self._no_errors else TestResult.FAILED
        self._manager.close_test(status)

    def encountered_error(self):
        self._no_errors = False


class RegressionContext(object):
    def __init__(self, manager: RegressionManager, output_folder: pathlib.Path):
        self._manager = manager
        self._test_case = None
        self._output_folder = output_folder

    def open_test(self, test_case: unittest.TestCase, test_name: str) -> TestContext:
        self._test_case = test_case
        return TestContext(test_name, self._manager, self._output_folder)

    @contextmanager
    def open_subtest(self, test_fn_name: str, line):
        with (
            self._test_case.subTest(line=line) as subtest,
            TimestampedContext(f"{test_fn_name} {line}") as timestamp_ctx,
        ):
            yield (subtest, timestamp_ctx)
