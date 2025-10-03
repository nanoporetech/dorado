import unittest

from data_paths import ROOT_DIR, REFERENCE_FOLDER
from tetra.compare_platforms import PlatformCompare, SpecialCase, compare_platforms


class TestCompareData(unittest.TestCase):
    def test_compare_platforms(self):
        """
        This compares test results from different platforms. As new tests are
        added to the project, we may need to add new `special_cases` and/or
        `exclude_columns` fields of the `PlatformCompare` objects.
        """
        REF_DIR = REFERENCE_FOLDER.parent

        # The default tolerance of 4% addresses the fact that basecalling results
        # differ quite considerably between the Linux, OSX-ARM, and Orin platforms.
        # A 10% tolerance is needed for the RNA tests, as the qscores in particular
        # differ between platforms by a considerable amount.
        DEFAULT_TOLERANCE = 0.05
        RNA_TOLERANCE = 0.10
        rna_exception = SpecialCase(pattern="RNA", tolerance=RNA_TOLERANCE)

        # Windows CUDA gives nearly identical results to Linux CUDA, so we use a
        # tighter tolerance of 0.5%.
        WINDOWS_TOLERANCE = 0.005

        # Platform comparison for barcoding only works between linux_cuda and windows,
        # due to the fact that even tiny differences in basecalling can result in
        # different classifications.
        exclude_barcoding = SpecialCase(pattern="barcoding", tolerance=None)

        # The RNA alignment tests give different alignment results for a couple of
        # reads, due to slightly different basecalls. We should consider removing those
        # reads from the dataset. For now this exclusion deals with them.
        exclude_align_rna = SpecialCase(pattern="alignment/RNA", tolerance=None)

        # Modified basecalling tests give differing results on different platforms, so
        # we can only really compare linux_cuda and windows.
        exclude_modbase = SpecialCase(pattern="modified_bases", tolerance=None)

        platforms_to_compare = [
            PlatformCompare(
                reference_path=REF_DIR,
                platform_1="linux",
                platform_2="windows",
                platform_tolerance=WINDOWS_TOLERANCE,
                special_cases=[],
                exclude_columns=["passes_filtering"],
            ),
            PlatformCompare(
                reference_path=REF_DIR,
                platform_1="linux",
                platform_2="osx_arm",
                platform_tolerance=DEFAULT_TOLERANCE,
                special_cases=[
                    exclude_barcoding,
                    exclude_align_rna,
                    exclude_modbase,
                    rna_exception,
                ],
                exclude_columns=None,
            ),
            PlatformCompare(
                reference_path=REF_DIR,
                platform_1="linux",
                platform_2="orin",
                platform_tolerance=DEFAULT_TOLERANCE,
                special_cases=[
                    exclude_align_rna,
                    exclude_modbase,
                    rna_exception,
                ],
                exclude_columns=None,
            ),
        ]
        for comparison in platforms_to_compare:
            with self.subTest(
                line=f"folder={REF_DIR}, platform1={comparison.platform_1}, platform2={comparison.platform_2}, tolerance={comparison.platform_tolerance}"
            ):
                result = compare_platforms(comparison)
                if result:
                    result.insert(
                        0,
                        (
                            f"\n------  {comparison.platform_1} <-> {comparison.platform_2}  ------\n"
                        ),
                    )
                    self.fail("\n".join(result))
