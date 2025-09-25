import json
import pathlib
import typing
from dataclasses import asdict, dataclass
from enum import Enum


class TestResult(str, Enum):
    ABORTED = "aborted"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass(slots=True)
class TestFile:
    name: str
    subfolder: str | None
    exists_in_ref: bool
    exists_in_output: bool
    matches: bool | None
    validation_passed: bool | None


@dataclass(slots=True)
class TestData:
    name: str
    test_folder: str
    subfolders: set[str]
    test_files: list[TestFile]
    result: TestResult | None


@dataclass(slots=True)
class ManagerData:
    tests: list[TestData]


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class RegressionManager(object):
    def __init__(self):
        self._results = ManagerData([])
        self._current_test = None
        self._main_output_folder = None

    @property
    def is_test_open(self) -> bool:
        return self._current_test is not None

    @property
    def results(self) -> ManagerData:
        return self._results

    @property
    def main_output_folder(self) -> pathlib.Path | None:
        return self._main_output_folder

    def set_main_output_folder(self, output: pathlib.Path):
        """
        Set optional main output folder.
        This is only necessary if you want to use the `collect_files` method.
        """
        self._main_output_folder = output

    def open_test(self, name: str, subfolder: str):
        """
        This registers the test name, and assigns to it a subfolder.
        If another test is currently open, this will close it with the
        ABORTED result state.
        """
        if self._current_test is not None:
            self.close_current_test(TestResult.ABORTED)
        self._current_test = TestData(
            name=name,
            test_folder=subfolder,
            subfolders=set(),
            test_files=[],
            result=None,
        )

    def add_test_files(self, test_files: list[TestFile]):
        """
        Use this to register test files. Each entry should be a TestFile object.
        Use `None` for the `subfolder` field if the file should be in the main
        test folder.

        If either the reference or output file doesn't exist, they should still be
        registered here, and the entry in `test_files` should have the `matches`
        field set to `None`.

        If the file is not subject to specification validation, then the
        `validation_passed` field should be `None`.
        """
        if self._current_test is None:
            raise RuntimeError("Can't add test files to a test that isn't open.")
        for test_file in test_files:
            if test_file.subfolder is not None:
                self._current_test.subfolders.add(test_file.subfolder)
        self._current_test.test_files.extend(test_files)

    def collect_files(
        self, subfolder: str | None, expected: dict[str, int]
    ) -> list[TestFile]:
        """
        Automatically scan the output folder of an executable that has been run,
        for expected output files.

        You must call `set_main_output_folder` before calling this method.

        The `subfolder` argument can be used if the output folder of the executable
        is a subfolder of the folder of the currently open test. Use `None` if the
        output folder of the executable is just the folder of the currently open
        test.

        For each expected type, you need to specify how many files of that type you
        expect to see. This method will scan through the output folder of the
        executable recursively for files of the specified type.

        If the number of files doesn't match the expectation, this method will raise
        an exception. If you choose to handle that exception, you MUST make sure
        that the `TestData` object is marked as either ABORTED or FAILED.

        The returned list of TestFile objects can then be passed through the
        `DataChecker` class, and finally added to the open test.
        """
        if self._main_output_folder is None:
            raise Exception(
                "Error: The main output folder must be set before calling `collect_files`."
            )
        test_folder = self._main_output_folder / self._current_test.test_folder
        scan_folder = test_folder
        if subfolder is not None:
            scan_folder = test_folder / subfolder
        found_files = []
        for file_type, expected_files in expected.items():
            full_file_paths = list(scan_folder.glob(f"**/*.{file_type}"))
            if len(full_file_paths) != expected_files:
                message = (
                    f'Error: For test "{self._current_test.name}", found {len(full_file_paths)} '
                    f"files of type {file_type}, but expected {expected_files} files."
                )
                raise Exception(message)
            found_files.extend([f.relative_to(test_folder) for f in full_file_paths])
        test_files = []
        for file in found_files:
            if file.name == str(file):
                file_subfolder = None
            else:
                file_subfolder = str(file.parent)
            test_files.append(
                TestFile(
                    name=file.name,
                    subfolder=file_subfolder,
                    exists_in_ref=False,
                    exists_in_output=False,
                    matches=None,
                    validation_passed=None,
                )
            )
        return test_files

    def close_test(self, result: TestResult):
        """
        Once all files have been registered for a test, close the test.
        The `result` argument should be `COMPLETED` if the test completed without
        errors that would result in the output files either being not present or
        incorrect. The `FAILED` result should be used if such errors did occur, and
        the `ABORTED` result should be used if the test was unable to complete at all.
        """
        if self._current_test is None:
            raise RuntimeError("Can't close a test that isn't open.")
        self._current_test.result = result
        self._results.tests.append(self._current_test)
        self._current_test = None

    def write_to_file(self, outfile: pathlib.Path):
        """
        Writes out the contents of the class to a file, in JSON format.
        """
        if self._current_test is not None:
            self.close_current_test(TestResult.ABORTED)
        json_data = asdict(self._results)
        outfile.write_text(
            json.dumps(json_data, sort_keys=True, indent=4, cls=SetEncoder)
        )

    @classmethod
    def read_from_file(cls, infile: pathlib.Path) -> "RegressionManager":
        """
        Read the contents of test results that have been serialized with
        `RegressionManager.write_to_file`.
        """
        if not infile.exists():
            raise FileNotFoundError(f"Test results file '{infile}' not found.")
        json_data = json.loads(infile.read_text())

        # Validate that the json contains what we expect.
        expected_fields = {"tests": list}
        if missing_fields := set(expected_fields.keys()) - set(json_data.keys()):
            raise Exception(f"Failed to parse file '{infile}'. '{missing_fields=}'.")

        for key, dtype in expected_fields.items():
            if not isinstance(json_data[key], dtype):
                raise Exception(
                    f"Failed to parse '{infile}'. Field '{key}' must be a '{dtype}'."
                )

        manager = cls()
        manager._results.tests.extend(cls.parse_test(t) for t in json_data["tests"])
        return manager

    # Internal methods below. These are not meant to be called directly.
    @classmethod
    def parse_test(cls, test_field) -> TestData:
        expected_fields = {
            "name": str,
            "test_folder": str,
            "subfolders": list,
            "test_files": list,
            "result": str,
        }
        if missing_fields := set(expected_fields.keys()) - set(test_field.keys()):
            raise Exception(f"Failed to parse test entry. '{missing_fields=}'.")

        for key, dtype in expected_fields.items():
            if not isinstance(test_field[key], dtype):
                raise Exception(
                    f"Failed to parse test entry. Field '{key}' must be a '{dtype}'."
                )

        test_field["test_folder"] = cls.sanitize_folder_name(test_field["test_folder"])
        test_field["subfolders"] = [
            cls.sanitize_folder_name(folder) for folder in test_field["subfolders"]
        ]

        parsed_test_files = [cls.parse_test_file(f) for f in test_field["test_files"]]
        parsed_result = cls.parse_test_result(test_field["result"])

        return TestData(
            test_field["name"],
            test_field["test_folder"],
            set(test_field["subfolders"]),
            parsed_test_files,
            parsed_result,
        )

    @classmethod
    def parse_test_file(cls, test_file_field: dict[str, typing.Any]) -> TestFile:
        if not isinstance(test_file_field, dict):
            raise Exception(
                "Failed to parse 'test_files' entry of results file (must be a dict)."
            )
        # Meaning of field elements is (key, type, can-be-None).
        expected_fields = {
            "name": str,
            "subfolder": str | None,
            "exists_in_ref": bool,
            "exists_in_output": bool,
            "matches": bool | None,
            "validation_passed": bool | None,
        }
        for field, field_type in expected_fields.items():
            if field not in test_file_field:
                raise Exception(
                    f"Failed to parse 'test_files' field of results file. Field '{field}' not found."
                )
            json_field = test_file_field[field]
            if not isinstance(json_field, field_type):
                raise Exception(
                    f"Failed to parse 'test_files' field of results file. Field '{field}' has incorrect type, expected '{field_type}'."
                )

        test_file_field["subfolder"] = cls.sanitize_folder_name(
            test_file_field["subfolder"]
        )

        return TestFile(
            test_file_field["name"],
            test_file_field["subfolder"],
            test_file_field["exists_in_ref"],
            test_file_field["exists_in_output"],
            test_file_field["matches"],
            test_file_field["validation_passed"],
        )

    @classmethod
    def parse_test_result(cls, test_result_field: str | None) -> TestResult | None:
        if test_result_field is None:
            return None
        if isinstance(test_result_field, str):
            return TestResult(test_result_field)
        raise Exception("Failed to parse 'result' field of results file.")

    @classmethod
    def sanitize_folder_name(cls, folder_name: str | None) -> str | None:
        """
        If the file was saved on Windows, the folder names will have `\\` as delimiters.
        These need to be replaced with `/`, which should work on all platforms.
        """
        if folder_name is None:
            return None
        return folder_name.replace("\\", "/")
