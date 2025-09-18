import argparse
import pathlib
import shutil
from dataclasses import dataclass
from enum import Enum

from tetra.reformat_files import reformat_files
from tetra.regression_manager import ManagerData, RegressionManager, TestResult

"""
To update regression test reference data:
    1) Run the regression test(s) to produce expected output.
    2) Run the `check_and_update` function below.
    3) Confirm that the report gives the expected results.
    4) Add and commit the updated reference files.
"""

QUIET_MODE = False


def quiet_print(*args):
    if not QUIET_MODE:
        print(*args)


class RebaseResult(str, Enum):
    UNKNOWN = "Test results unknown"
    ALL_TESTS_PASSED = "All tests passed"
    CAN_BE_REBASED = "Folder can be rebased"
    CANNOT_BE_REBASED = "Folder cannot be rebased"


@dataclass(slots=True)
class PlatformFolder:
    name: str
    in_ref: bool
    in_output: bool
    errors: list[str]
    platform_result: RebaseResult
    files_to_remove: list[pathlib.Path] | None = None
    files_to_add: list[pathlib.Path] | None = None
    files_to_replace: list[pathlib.Path] | None = None


@dataclass(slots=True)
class RegressionSummary:
    platform_folders: list[PlatformFolder] | None
    full_result: RebaseResult


def inspect_output_files(
    reference_folder: pathlib.Path, output_folder: pathlib.Path, platform: str | None
) -> RegressionSummary:
    summary = RegressionSummary([], RebaseResult.UNKNOWN)

    # Compare platform folders.
    if platform is None:
        ref_platform_folders = {
            x.name for x in reference_folder.iterdir() if x.is_dir()
        }
        output_platform_folders = {
            x.name for x in output_folder.iterdir() if x.is_dir()
        }
    else:
        ref_platform_folders = [platform]
        output_platform_folders = [platform]
    for folder in ref_platform_folders:
        if folder in output_platform_folders:
            folder_details = inspect_platform(reference_folder, output_folder, folder)
        else:
            folder_details = PlatformFolder(
                name=folder,
                in_ref=True,
                in_output=False,
                errors=["Output folder for platform is missing."],
                platform_result=RebaseResult.CANNOT_BE_REBASED,
            )
        summary.platform_folders.append(folder_details)

    # Check for output platform folders missing from ref folder.
    for folder in output_platform_folders:
        if folder not in ref_platform_folders:
            folder_details = inspect_platform(None, output_folder, folder)
            summary.platform_folders.append(folder_details)

    for platform_data in summary.platform_folders:
        if platform_data.platform_result == RebaseResult.ALL_TESTS_PASSED:
            if summary.full_result == RebaseResult.UNKNOWN:
                summary.full_result = RebaseResult.ALL_TESTS_PASSED
        elif platform_data.platform_result == RebaseResult.CAN_BE_REBASED:
            if summary.full_result != RebaseResult.CANNOT_BE_REBASED:
                summary.full_result = RebaseResult.CAN_BE_REBASED
        else:
            summary.full_result = RebaseResult.CANNOT_BE_REBASED
    return summary


def check_for_json_summary(
    output_folder: pathlib.Path, folder: str
) -> tuple[PlatformFolder, ManagerData | None]:
    platform_output_path = output_folder / folder
    json_result_file = platform_output_path / "regression_test_results.json"

    try:
        manager = RegressionManager.read_from_file(json_result_file)
        test_results = manager.results
    except Exception as ex:
        platform_details = PlatformFolder(
            name=folder,
            in_ref=True,
            in_output=True,
            errors=[str(ex)],
            platform_result=RebaseResult.CANNOT_BE_REBASED,
        )
        return platform_details, None

    platform_details = PlatformFolder(
        name=folder,
        in_ref=True,
        in_output=True,
        errors=[],
        platform_result=RebaseResult.UNKNOWN,
    )
    return platform_details, test_results


def clear_file_and_path_fields(platform_details: PlatformFolder):
    platform_details.files_to_remove = None
    platform_details.files_to_add = None
    platform_details.files_to_replace = None


def inspect_platform(
    reference_folder: pathlib.Path | None, output_folder: pathlib.Path, folder: str
) -> PlatformFolder:
    platform_details, test_results = check_for_json_summary(output_folder, folder)
    if reference_folder is None:
        platform_details.in_ref = False
    if test_results is None:
        return platform_details

    # Use the regression test summary to determine what updates are needed (or if they are not possible).
    platform_details.files_to_add = []
    platform_details.files_to_replace = []
    expected_files = set()
    for test in test_results.tests:
        if test.result != TestResult.COMPLETED:
            platform_details.platform_result = RebaseResult.CANNOT_BE_REBASED
            platform_details.errors.append(
                "Test(s) were aborted or failed with errors that prevent rebasing."
            )
            clear_file_and_path_fields(platform_details)
            return platform_details
        test_folder = pathlib.Path(test.test_folder)
        for test_file in test.test_files:
            # We need to store the path of the file relative to the platform folder.
            file = test_folder
            if test_file.subfolder is not None:
                file /= test_file.subfolder
            file /= test_file.name
            expected_files.add(file)
            full_file_path = output_folder / folder / file
            if not full_file_path.exists() or not full_file_path.is_file():
                # File should exist, otherwise test.result would not be COMPLETED.
                # But something could go wrong with file system.
                platform_details.platform_result = RebaseResult.CANNOT_BE_REBASED
                platform_details.errors.append(
                    f"Output file `{full_file_path}` not found."
                )
                clear_file_and_path_fields(platform_details)
                return platform_details
            if reference_folder is None:
                platform_details.files_to_add.append(file)
            else:
                full_ref_path = reference_folder / folder / file
                if full_ref_path.exists():
                    if not test_file.matches:
                        platform_details.files_to_replace.append(file)
                else:
                    platform_details.files_to_add.append(file)

    # Now check for files to be removed.
    if reference_folder is not None:
        files_to_remove = []
        ref_platform_folder = reference_folder / folder
        all_ref_files = [
            f.relative_to(ref_platform_folder)
            for f in ref_platform_folder.glob("**/*")
            if f.is_file()
        ]
        for ref_file in all_ref_files:
            if ref_file not in expected_files:
                files_to_remove.append(ref_file)
        if files_to_remove:
            platform_details.files_to_remove = files_to_remove

    if (
        platform_details.files_to_add
        or platform_details.files_to_replace
        or platform_details.files_to_remove
    ):
        platform_details.platform_result = RebaseResult.CAN_BE_REBASED
    else:
        platform_details.platform_result = RebaseResult.ALL_TESTS_PASSED

    return platform_details


def write_report(summary: RegressionSummary, output_folder: pathlib.Path):
    report = output_folder / "regression_summary.txt"
    with report.open("w") as out:
        if summary.full_result == RebaseResult.ALL_TESTS_PASSED:
            out.write("All regression tests passed. Nothing to rebase.\n")
        elif summary.full_result == RebaseResult.CANNOT_BE_REBASED:
            out.write("Cannot rebase reference files, due to the following errors:\n")
            for platform_folder in summary.platform_folders:
                if platform_folder.errors:
                    out.write(f"Platform folder: {platform_folder.name}\n")
                    for line in platform_folder.errors:
                        out.write(f"ERROR: {line}\n")
        else:
            out.write("Rebasing can be performed.\n")
            for platform_folder in summary.platform_folders:
                if platform_folder.platform_result == RebaseResult.CAN_BE_REBASED:
                    out.write(f"Platform folder: {platform_folder.name}\n")
                    if platform_folder.files_to_add:
                        out.write("  Files to be added to reference folder:\n")
                        for file in platform_folder.files_to_add:
                            out.write(f"    {file}\n")
                    if platform_folder.files_to_replace:
                        out.write("  Files to be replaced in reference folder:\n")
                        for file in platform_folder.files_to_replace:
                            out.write(f"    {file}\n")
                    if platform_folder.files_to_remove:
                        out.write("  Files to be removed from reference folder:\n")
                        for file in platform_folder.files_to_remove:
                            out.write(f"    {file}\n")
    quiet_print("Regression test summary written to:", report)


def update_reference_folder(
    reference_folder: pathlib.Path,
    output_folder: pathlib.Path,
    summary: RegressionSummary,
):
    for platform_folder in summary.platform_folders:
        if platform_folder.platform_result == RebaseResult.CAN_BE_REBASED:
            if platform_folder.files_to_add:
                for file in platform_folder.files_to_add:
                    ref_file = reference_folder / platform_folder.name / file
                    out_file = output_folder / platform_folder.name / file
                    ref_dir = ref_file.parent
                    if not ref_dir.exists():
                        ref_dir.mkdir(parents=True)
                        quiet_print("Created missing folder:", ref_dir)
                    shutil.copyfile(out_file, ref_file)
                    quiet_print("Added reference file:", ref_file)
            if platform_folder.files_to_replace:
                for file in platform_folder.files_to_replace:
                    ref_file = reference_folder / platform_folder.name / file
                    out_file = output_folder / platform_folder.name / file
                    shutil.copyfile(out_file, ref_file)
                    quiet_print("Replaced reference file:", ref_file)
            if platform_folder.files_to_remove:
                for file in platform_folder.files_to_remove:
                    ref_file = reference_folder / platform_folder.name / file
                    ref_file.unlink()
                    quiet_print("Removed reference file:", ref_file)


def run_update(
    test_folder: pathlib.Path, platform: str | None, update_refs: bool, quiet_mode: bool
):
    global QUIET_MODE
    QUIET_MODE = quiet_mode
    reference_folder = test_folder / "ref"
    output_folder = test_folder / "output"
    quiet_print("Reference folder:", reference_folder)
    quiet_print("Output folder:", output_folder)
    quiet_print("Formatting output files for deterministic testing.")
    reformat_files(output_folder)
    quiet_print("Inspecting regression test results to determine required changes.")
    summary = inspect_output_files(reference_folder, output_folder, platform)
    if summary.full_result == RebaseResult.ALL_TESTS_PASSED:
        quiet_print("All tests passed. No rebase is needed.")
    elif summary.full_result == RebaseResult.CANNOT_BE_REBASED:
        quiet_print("Rebase cannot be performed due to errors. See report for details.")
    write_report(summary, output_folder)
    if update_refs and summary.full_result == RebaseResult.CAN_BE_REBASED:
        quiet_print("Rebasing reference files.")
        update_reference_folder(reference_folder, output_folder, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_folder")
    parser.add_argument("--update_refs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    test_folder = pathlib.Path(args.test_folder)
    run_update(
        test_folder=test_folder,
        platform=None,
        update_refs=args.update_refs,
        quiet_mode=args.quiet,
    )
