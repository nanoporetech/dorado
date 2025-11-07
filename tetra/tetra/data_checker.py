import pathlib
from collections import defaultdict

from tetra.data_frame_utils import check_pandas_equal, load_sorted_data_frame
from tetra.regression_manager import TestData
from tetra.sequence_utils import compare_alignment_files, compare_fastq_files

try:
    from tetra.output_validation import setup_validator
except ImportError:
    print("ont-output-specification-validator not available.")

    def setup_validator(*_):
        return None, None


class DataChecker(object):
    def __init__(
        self, ref_folder: str, output_folder: str, tolerance_factor: float = 1.0
    ):
        """
        The constructor will check for the existence of the output folder, and
        raise an exception if it is missing.
        """
        self.tolerance_factor = tolerance_factor
        self.ref_path = pathlib.Path(ref_folder)
        self.output_path = pathlib.Path(output_folder)
        if not self.output_path.exists() or not self.output_path.is_dir():
            raise Exception(f"Expected output folder `{output_folder}` does not exist.")

    def check_test_data(self, data: TestData, validation_options: dict | None) -> dict:
        """
        This will compare the expected reference and output files indicated in
        `data`, and if `validation_options` is not `None`, it will validate the
        output files.

        This will first check for the existence of the expected files, and update
        the fields in the individual `TestFile` entries. For each test file,
        the function will compare the output and reference files. Validation is
        then performed on each output file subject to the specified validation
        options. The comparison and validation results will be added to `data`.

        If any errors occur, they will be reported in the returned dict. It will
        contain two entries: "comparison" and "validation". Each will contain a
        list of strings which can be reported by the calling regression test as
        needed.
        """
        test_ref_path = self.ref_path / data.test_folder
        test_output_path = self.output_path / data.test_folder
        results = {"comparison": [], "validation": []}
        validator, options = setup_validator(validation_options)
        if validator is None and validation_options is not None:
            results["validation"].append(
                "Warning. Validation was requested, but is not enabled."
            )
        for test_file in data.test_files:
            ref_file_path = test_ref_path
            if test_file.subfolder is not None:
                ref_file_path /= test_file.subfolder
            ref_file_path /= test_file.name
            if ref_file_path.exists() and ref_file_path.is_file():
                test_file.exists_in_ref = True
            else:
                test_file.exists_in_ref = False
                results["comparison"].append(
                    f"Reference file `{ref_file_path}' does not exist."
                )

            output_file_path = test_output_path
            if test_file.subfolder is not None:
                output_file_path /= test_file.subfolder
            output_file_path /= test_file.name
            if output_file_path.exists() and output_file_path.is_file():
                test_file.exists_in_output = True
            else:
                test_file.exists_in_output = False
                results["comparison"].append(
                    f"Output file `{output_file_path}' does not exist."
                )

            # If both the reference and output file exist, compare them.
            if test_file.exists_in_output and test_file.exists_in_ref:
                errors = DataChecker.compare_files(
                    ref_file_path, output_file_path, self.tolerance_factor
                )
                if errors is None:
                    test_file.matches = True
                else:
                    test_file.matches = False
                    results["comparison"].append(errors)
            else:
                test_file.matches = None

            # Validate the output file, if appropriate.
            if test_file.exists_in_output and validator is not None:
                result, errors = DataChecker.validate_file(
                    output_file_path, validator, options
                )
                test_file.validation_passed = result
                if errors is not None:
                    results["validation"].append(errors)
            else:
                test_file.validation_passed = None

        return results

    # Internal methods below. These are not meant to be called directly.
    @classmethod
    def compare_files(
        cls, ref_file: pathlib.Path, output_file: pathlib.Path, tolerance_factor: float
    ) -> str | None:
        regression_diffs = None
        extension = ref_file.suffix
        if output_file.suffix != extension:
            regression_diffs = "File extensions do not match."
        elif extension == ".txt":
            try:
                actual_df = load_sorted_data_frame(output_file)
                expected_df = load_sorted_data_frame(ref_file)
                regression_diffs = check_pandas_equal(
                    actual_df, expected_df, tolerance_factor
                )
            except Exception:
                return f"check_pandas_equal failed for {output_file}, {ref_file}"
            if regression_diffs == "":
                regression_diffs = None
        elif extension in [".sam", ".bam"]:
            result, message = compare_alignment_files(output_file, ref_file)
            if not result:
                regression_diffs = message
        elif extension == ".fastq":
            result, message = compare_fastq_files(output_file, ref_file)
            if not result:
                regression_diffs = message
        else:
            regression_diffs = "Unrecognised file extension"

        if regression_diffs is not None:
            return f"Actual file: {output_file}\nReference File: {ref_file}\n{regression_diffs}"
        return None

    @classmethod
    def validate_file(
        cls, output_file: pathlib.Path, validator, options
    ) -> tuple[bool | None, str | None]:
        ext_map = defaultdict(bool)
        ext_map.update(
            {
                ".fastq": options.do_fastq,
                ".bam": options.do_bam,
                ".txt": options.do_summary,
            }
        )
        extension = output_file.suffix

        if ext_map[extension]:
            return validator.validate_file(output_file)
        return None, None

    @classmethod
    def make_error_message(cls, checker_results: dict) -> str | None:
        error_message = None
        if checker_results["comparison"]:
            error_message = "\n".join(checker_results["comparison"])
        if checker_results["validation"]:
            if error_message is None:
                error_message = ""
            else:
                error_message += "\n"
            error_message += "\n".join(checker_results["validation"])
        return error_message
