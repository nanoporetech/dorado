import pathlib
from dataclasses import dataclass

from tetra.data_frame_utils import load_sorted_data_frame
from tetra.error_display import pretty_print_diffs


@dataclass(slots=True)
class SpecialCase:
    pattern: str
    tolerance: float | None


@dataclass(slots=True)
class PlatformCompare:
    reference_path: pathlib.Path
    platform_1: str
    platform_2: str
    platform_tolerance: float
    special_cases: list[SpecialCase] | None = None
    exclude_columns: list[str] | None = None


def check_file(file: pathlib.Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern in str(file):
            return False
    return True


def compare_platforms(platform_compare: PlatformCompare):
    """
    We check all .txt files within the platform folders. Specific filenames
    and/or folders can be omitted by using special cases in the PlatformCompare
    object passed into this function.
    """
    failed_comparisons = []
    platform_1 = platform_compare.platform_1
    platform_2 = platform_compare.platform_2
    platform_folder_1 = platform_compare.reference_path / platform_1
    platform_folder_2 = platform_compare.reference_path / platform_2
    platform_file_paths_1 = platform_folder_1.glob("**/*.txt")
    platform_file_paths_2 = platform_folder_2.glob("**/*.txt")

    all_platform_files_1 = [
        f.relative_to(platform_folder_1) for f in platform_file_paths_1
    ]
    all_platform_files_2 = [
        f.relative_to(platform_folder_2) for f in platform_file_paths_2
    ]

    # Remove any files that should be skipped.
    skip_patterns = []
    if platform_compare.special_cases:
        for case in platform_compare.special_cases:
            # A value of None for a special tolerance means that the file should be skipped.
            if case.tolerance is None:
                skip_patterns.append(case.pattern)
    platform_files_1 = [
        file for file in all_platform_files_1 if check_file(file, skip_patterns)
    ]
    platform_files_2 = [
        file for file in all_platform_files_2 if check_file(file, skip_patterns)
    ]

    # First, make sure every file is present in both platforms.
    files_in_both = set()
    for file in platform_files_1:
        if file not in platform_files_2:
            failed_comparisons.append(
                f"\n{file}\nFile found in '{platform_1}' but missing from '{platform_2}'"
            )
        else:
            files_in_both.add(file)
    for file in platform_files_2:
        if file not in platform_files_1:
            failed_comparisons.append(
                f"\n{file}\nFile found in '{platform_2}' but missing from '{platform_1}'"
            )

    for file in files_in_both:
        tolerance = platform_compare.platform_tolerance
        # We may need to apply a special tolerance to files whose paths contain
        # particular patterns.
        if platform_compare.special_cases:
            for case in platform_compare.special_cases:
                if case.pattern in str(file):
                    tolerance = case.tolerance

        try:
            frame1 = load_sorted_data_frame(platform_folder_1 / file)
            frame2 = load_sorted_data_frame(platform_folder_2 / file)

            if platform_compare.exclude_columns:
                for column in platform_compare.exclude_columns:
                    if column in frame1.columns:
                        frame1.drop(columns=[column], inplace=True)
                        frame2.drop(columns=[column], inplace=True)

            diffs = pretty_print_diffs(frame1, frame2, tolerance)
            if diffs:
                failed_comparisons.append(f"\n{file}\n{diffs}")
        except Exception as ex:
            failed_comparisons.append(f"\n{file}\nError parsing files: {ex}")
    return failed_comparisons
