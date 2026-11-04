#include "secondary/common/variant.h"
#include "secondary/consensus/variant_normalization.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dorado::secondary::tests {

#define TEST_GROUP "[Secondary][VariantNormalization]"

CATCH_TEST_CASE("find_common_prefix_suffix_len", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref;
        std::vector<std::string> alts;
        std::pair<int64_t, int64_t> expected_allow_empty;
        std::pair<int64_t, int64_t> expected_non_empty;
    };

    // Expected pairs are {prefix, suffix}; the suffix takes precedence when they overlap.
    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{"Empty inputs", "", {}, {0, 0}, {0, 0}},
        TestCase{"Empty reference allele", "", {"AC"}, {0, 0}, {0, 0}},
        TestCase{"Empty alternate allele", "AC", {""}, {0, 0}, {0, 0}},
        TestCase{"One of several alternate alleles is empty", "AC", {"AT", ""}, {0, 0}, {0, 0}},
        TestCase{"No alternate alleles", "ACGT", {}, {0, 0}, {0, 0}},
        TestCase{"No common ends", "AC", {"GT"}, {0, 0}, {0, 0}},
        TestCase{"Common prefix only", "AAC", {"AAT"}, {2, 0}, {2, 0}},
        TestCase{"Common suffix only", "CAA", {"TAA"}, {0, 2}, {0, 2}},
        TestCase{"Common prefix and suffix", "AACGG", {"AATGG"}, {2, 2}, {2, 2}},
        TestCase{"Identical single bases", "A", {"A"}, {0, 1}, {0, 0}},
        TestCase{"Identical alleles prefer suffix trimming", "ACGT", {"ACGT"}, {0, 4}, {0, 3}},
        TestCase{"Shortest allele is a prefix", "AC", {"ACG"}, {2, 0}, {1, 0}},
        TestCase{"Shortest allele is a suffix", "AC", {"TAC"}, {0, 2}, {0, 1}},
        TestCase{"Prefix and suffix compete for the anchor", "ATA", {"A"}, {0, 1}, {0, 0}},
        TestCase{"Prefix and suffix together consume the shortest allele", "ATG", {"AG"}, {1, 1}, {0, 1}},
        TestCase{"Reference is longer than alternate", "AAAA", {"AA"}, {0, 2}, {0, 1}},
        TestCase{"Duplicate alternate alleles", "ACG", {"ATG", "ATG"}, {1, 1}, {1, 1}},
        TestCase{"All alternate alleles constrain the prefix", "AACG", {"AATG", "ATCG"}, {1, 1}, {1, 1}},
        TestCase{"All alternate alleles constrain the suffix", "ACGG", {"ATGG", "ACAG"}, {1, 1}, {1, 1}},
        TestCase{"Shortest alternate is not first", "AAAA", {"AAA", "A", "AA"}, {0, 1}, {0, 0}},
        TestCase{"One alternate prevents all trimming", "AC", {"AT", "GC"}, {0, 0}, {0, 0}},
    }));
    // clang-format on

    const bool non_empty = GENERATE(false, true);
    CATCH_INFO("Test name: " << test_case.name);
    CATCH_CAPTURE(non_empty);

    const auto expected = non_empty ? test_case.expected_non_empty : test_case.expected_allow_empty;
    CATCH_CHECK(find_common_prefix_suffix_len(test_case.ref, test_case.alts, non_empty) ==
                expected);
}

CATCH_TEST_CASE("normalize_variant_vs_ref_core within the bounds of the neighboring variants",
                TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref_seq;
        int64_t prev_variant_end;
        int64_t next_variant_start;
        Variant variant;
        Variant expected_variant;
        bool expected_changed;
        bool allow_ambig_ref;
    };

    // Coordinates are zero-based. Neighboring records bound the interval [prev_end, next_start), which
    // is the window of ref_seq passed to the normalization.
    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "SNP is unchanged", "ACGT", 0, 4,
            Variant{0, 1, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "MNP without common ends is unchanged", "ACGT", 0, 4,
            Variant{0, 1, "CG", {"TA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "CG", {"TA"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Trim common prefix", "TACGTA", 0, 6,
            Variant{0, 1, "AC", {"AT"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Trim common suffix", "TACGTA", 0, 6,
            Variant{0, 2, "CG", {"TG"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Trim both ends", "TACGTA", 0, 6,
            Variant{0, 1, "ACG", {"ATG"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Trim several bases from both ends", "AACCGGG", 0, 7,
            Variant{0, 0, "AACCGGG", {"AATTGGG"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "CC", {"TT"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Prefix trimming is allowed at the previous variant boundary", "TACGTA", 1, 6,
            Variant{0, 1, "ACG", {"ATG"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Suffix trimming is allowed at chromosome start", "ACGT", 0, 4,
            Variant{0, 0, "AC", {"TC"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Trim both ends across all alternate alleles", "TACGTA", 0, 6,
            Variant{0, 1, "ACG", {"ATG", "AGG"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "C", {"T", "G"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "One alternate prevents suffix trimming", "TACGTA", 0, 6,
            Variant{0, 1, "ACG", {"ATG", "ACC"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "CG", {"TG", "CC"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Deletion reaches chromosome start", "AAAAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Insertion at last reference base reaches chromosome start", "AAAAAA", 0, 6,
            Variant{0, 5, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Deletion already at chromosome start retains its anchor", "AAAA", 0, 4,
            Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Insertion already at chromosome start retains its anchor", "AAAA", 0, 4,
            Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        // Shifting through a tandem repeat rotates the allele, not just its position.
        TestCase{
            "Deletion rotates through a tandem repeat", "ATATATAT", 0, 8,
            Variant{0, 3, "TAT", {"T"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "ATA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Insertion rotates through a tandem repeat", "ATATATAT", 0, 8,
            Variant{0, 3, "T", {"TAT"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {"ATA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Deletion rotates through a triplet repeat", "CAGCAGCAG", 0, 9,
            Variant{0, 4, "AGCA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "CAGC", {"C"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Insertion rotates through a triplet repeat", "CAGCAGCAG", 0, 9,
            Variant{0, 4, "A", {"AGCA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "C", {"CAGC"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Mixed-length alleles retain their order", "AAAAAA", 0, 6,
            Variant{0, 3, "AA", {"AAA", "A", "AAAA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AA", {"AAA", "A", "AAAA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "SNP allele prevents shifting a multiallelic insertion", "CAAAAA", 0, 6,
            Variant{0, 3, "A", {"AA", "T"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 3, "A", {"AA", "T"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Deletion stops at a non-repeated reference base", "GATATATC", 0, 8,
            Variant{0, 3, "ATA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "GAT", {"G"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Deletion stops at previous variant end", "AAAAAAAA", 2, 8,
            Variant{0, 5, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Insertion stops at previous variant end", "AAAAAAAA", 2, 8,
            Variant{0, 5, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Adjacent deletion cannot move left", "AAAAAAAA", 2, 8,
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Insertion on a single-base chromosome", "A", 0, 1,
            Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Empty alternate allele is left padded and shifted", "AAAAAA", 0, 6,
            Variant{0, 3, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "One empty alternate requires padding all alleles", "CAAAT", 0, 5,
            Variant{0, 3, "A", {"", "T"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A", "AT"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        // Use the next reference base when no left anchor is available.
        TestCase{
            "Empty alternate at chromosome start is right padded", "AAAA", 0, 4,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Empty alternate at previous variant end is right padded", "AAAAAA", 2, 6,
            Variant{0, 2, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Normalization preserves variant metadata", "AAAAAAA", 0, 7,
            Variant{7, 5, "AA", {"A"}, Filters::LowQual, {{"DP", "12"}}, 42.0f,
                    {{"GT", "0/1"}, {"AD", "7,5"}}, 5, 7},
            Variant{7, 0, "AA", {"A"}, Filters::LowQual, {{"DP", "12"}}, 42.0f,
                    {{"GT", "0/1"}, {"AD", "7,5"}}, 5, 7},
            true,
            false,
        },
        TestCase{
            "Ambiguous reference base stops shifting when disallowed", "ANAAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Ambiguous reference base can be used as an anchor when allowed", "ANAAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "NA", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            true,
        },
        TestCase{
            "Insertion stops at an IUPAC ambiguity code", "ARAAAA", 0, 6,
            Variant{0, 4, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Insertion can use an IUPAC ambiguity code when allowed", "ARAAAA", 0, 6,
            Variant{0, 4, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "R", {"RA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            true,
        },
        TestCase{
            "Lowercase ambiguous reference base stops shifting", "AnAAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Lowercase ambiguous anchor is preserved when allowed", "AnAAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "nA", {"n"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            true,
        },
        TestCase{
            "Lowercase canonical reference base is a valid anchor", "AcAAAA", 0, 6,
            Variant{0, 4, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "c", {"cA"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Empty alternate is right padded when ambiguity blocks left padding", "ANAAAA", 0, 6,
            Variant{0, 2, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Empty alternate can be padded with an allowed ambiguous base", "ANAAAA", 0, 6,
            Variant{0, 2, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "NA", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            true,
        },
        // Only the last shift is reverted; earlier suffix trimming must be retained.
        TestCase{
            "Suffix trimming is retained when ambiguity blocks further shifting", "ANAAAAC", 0, 7,
            Variant{0, 3, "AAAC", {"AAC"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Right padding applies to every alternate allele", "ACGT", 0, 4,
            Variant{0, 0, "AC", {"", "T"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "ACG", {"G", "TG"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Only an allowed ambiguous right anchor is available", "AN", 0, 2,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AN", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            true,
        },
        // Padding must stay strictly before next_variant_start, including for a multi-base REF.
        TestCase{
            "Right padding can use the base just before the next variant", "ACGT", 0, 2,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AC", {"C"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Right padding cannot use the next variant's first base", "ACGT", 0, 1,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Neighbors on both sides prevent padding", "ACGT", 2, 3,
            Variant{0, 2, "G", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "G", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Next variant blocks padding after a multi-base deletion", "ACGT", 0, 2,
            Variant{0, 0, "AC", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AC", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        // An unpadded call is retained when neither side provides a permitted anchor.
        TestCase{
            "Deletion of the entire chromosome remains unpadded", "ACGT", 0, 4,
            Variant{0, 0, "ACGT", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "ACGT", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Deletion of a single-base chromosome remains unpadded", "A", 0, 1,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        TestCase{
            "Previous variant blocks the only deletion anchor", "ACGT", 3, 4,
            Variant{0, 3, "T", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 3, "T", {""}, {}, {}, 0.0f, {}, 0, 0},
            false,
            false,
        },
        // An ambiguous base is used as a last resort to pad an empty allele, even when disallowed.
        TestCase{
            "Disallowed ambiguous right anchor is used if it is the only anchor", "AN", 0, 2,
            Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "AN", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Disallowed ambiguous left anchor is used if it is the only anchor", "NA", 0, 2,
            Variant{0, 1, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 0, "NA", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Deletion flanked by ambiguous bases on both sides is padded on the left", "TNGCNT", 0, 6,
            Variant{0, 2, "GC", {""}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "NGC", {"N"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Repeat matching is case-sensitive", "aaaAAA", 0, 6,
            Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 2, "aA", {"a"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Lowercase alleles retain their case", "acgt", 0, 4,
            Variant{0, 0, "acg", {"atg", "agg"}, {}, {}, 0.0f, {}, 0, 0},
            Variant{0, 1, "c", {"t", "g"}, {}, {}, 0.0f, {}, 0, 0},
            true,
            false,
        },
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    const std::string_view window = std::string_view{test_case.ref_seq}.substr(
            test_case.prev_variant_end, test_case.next_variant_start - test_case.prev_variant_end);
    const auto [result_var, result_status] = normalize_variant_vs_ref_core(
            test_case.variant, window, test_case.prev_variant_end, test_case.allow_ambig_ref);
    CATCH_CHECK(result_var == test_case.expected_variant);
    const NormalizationStatus expected_status = test_case.expected_changed
                                                        ? NormalizationStatus::CHANGED
                                                        : NormalizationStatus::UNCHANGED;
    CATCH_CHECK(result_status == expected_status);
    CATCH_REQUIRE(test_case.ref_seq.substr(result_var.pos, result_var.ref.size()) ==
                  result_var.ref);
    CATCH_CHECK(result_var.pos >= test_case.prev_variant_end);
    CATCH_CHECK(variant_end(result_var) <= test_case.next_variant_start);
    CATCH_CHECK(normalize_variant_vs_ref_core(result_var, window, test_case.prev_variant_end,
                                              test_case.allow_ambig_ref)
                        .first == result_var);
    CATCH_REQUIRE(result_var.alts.size() == test_case.variant.alts.size());

    if (!is_reference_record(result_var)) {
        // Reconstruct each alternate chromosome independently of the normalization algorithm.
        const auto apply_allele = [&test_case](const Variant& var, const std::string& alt) {
            return test_case.ref_seq.substr(0, var.pos) + alt +
                   test_case.ref_seq.substr(var.pos + var.ref.size());
        };
        auto original_alt = test_case.variant.alts.cbegin();
        for (const std::string& alt : result_var.alts) {
            CATCH_CAPTURE(alt);
            CATCH_CHECK(apply_allele(result_var, alt) ==
                        apply_allele(test_case.variant, *original_alt));
            ++original_alt;
        }
    }
}

CATCH_TEST_CASE("normalize_variants_vs_ref", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref_seq;
        std::vector<Variant> variants;
        std::vector<Variant> expected;
        bool allow_ambig_ref;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{"Empty inputs", "", {}, {}, false},
        TestCase{"No variants on a non-empty chromosome", "ACGT", {}, {}, false},
        TestCase{
            "Single deletion reaches chromosome start", "AAAAAA",
            {Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Single insertion reaches chromosome start", "AAAAAA",
            {Variant{0, 4, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "SNPs retain their order and alleles", "ACGT",
            {Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "T", {"C"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "T", {"C"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        // Each variant is shifted left until it reaches the previous variants after their normalization.
        TestCase{
            "Three deletions stop at the previous normalized deletions", "AAAAAAAAAAAA",
            {Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 5, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 8, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Unpadded deletions acquire anchors and stop at the previous normalized deletion", "AAAAAAAAAA",
            {Variant{0, 3, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 7, "AA", {""}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AAA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Tandem-repeat deletions rotate up to the previous normalized deletion", "ATATATATATAT",
            {Variant{0, 3, "TAT", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 9, "TAT", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "ATA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "TAT", {"T"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Insertion stops after the reference anchor of the previous normalized insertion", "AAAAAAAAAA",
            {Variant{0, 3, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Deletion stops immediately after a SNP", "AAAAAAAAAA",
            {Variant{0, 1, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Insertion stops after the full reference span of a deletion", "AAAAAAAAAA",
            {Variant{0, 3, "AAA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 7, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AAA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Deletion can follow the single reference anchor of an insertion", "AAAAAAAAAA",
            {Variant{0, 3, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Multiallelic indel blocks the following deletion", "AAAAAAAAAA",
            {Variant{0, 3, "AA", {"A", "AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 7, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AA", {"A", "AAA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Already adjacent normalized variants are unchanged", "AAAAAA",
            {Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Trimming a preceding variant moves the boundary for the next variant", "CAAAAAAA",
            {Variant{0, 0, "CAA", {"CTA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 5, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Deletions in separate repeats normalize independently", "CAAAGTTT",
            {Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "TT", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "CA", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 4, "GT", {"G"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        // Reference records are kept as they are, and their spans block shifting of later variants.
        // Standalone reference records are covered by "Normalization passes reference records through unchanged".
        TestCase{
            "Reference records between variants are preserved and block shifting", "AAAAAAAAAA",
            {Variant{0, 0, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 7, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "AA", {}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        // A reference block can span more bases than its single reference allele.
        TestCase{
            "Deletion stops at the END of a reference block", "AAAAAAAAAA",
            {Variant{0, 1, "A", {"<*>"}, {}, {{"END", "5"}}, 0.0f, {}, 0, 0},
             Variant{0, 7, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "A", {"<*>"}, {}, {{"END", "5"}}, 0.0f, {}, 0, 0},
             Variant{0, 5, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Insertion stops at the end of a reference block specified by LEN", "AAAAAAAAAA",
            {Variant{0, 1, "A", {"<*>"}, {}, {}, 0.0f, {{"LEN", "4"}}, 0, 0},
             Variant{0, 8, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "A", {"<*>"}, {}, {}, 0.0f, {{"LEN", "4"}}, 0, 0},
             Variant{0, 5, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Ambiguous reference stops normalization when disallowed", "ANAAAA",
            {Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Ambiguous reference option is forwarded", "ANAAAA",
            {Variant{0, 4, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "NA", {"N"}, {}, {}, 0.0f, {}, 0, 0}}, true,
        },
        TestCase{
            "Right-padded deletion remains adjacent to a following SNP", "ACGT",
            {Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "G", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AC", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "G", {"T"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Unpadded deletion beside a SNP uses a right anchor", "ACGT",
            {Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "C", {""}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "CG", {"G"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Following SNP blocks right padding", "ACGT",
            {Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Following reference record blocks right padding", "ACGT",
            {Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "C", {"<*>"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "C", {"<*>"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
        TestCase{
            "Following SNP blocks padding after a multi-base deletion", "ACGT",
            {Variant{0, 0, "AC", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "G", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "AC", {""}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "G", {"T"}, {}, {}, 0.0f, {}, 0, 0}}, false,
        },
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    const auto result = normalize_variants_vs_ref(test_case.variants, test_case.ref_seq,
                                                  test_case.allow_ambig_ref);
    CATCH_CHECK(result == test_case.expected);
    CATCH_REQUIRE(result.size() == test_case.variants.size());

    int64_t prev_end = 0;  // End of the previous normalized variant.
    int64_t i = 0;
    for (const Variant& variant : result) {
        CATCH_CAPTURE(variant);
        const int64_t next_start = (i + 1 < std::ssize(test_case.variants))
                                           ? test_case.variants[i + 1].pos
                                           : std::ssize(test_case.ref_seq);
        CATCH_CHECK(variant.pos >= prev_end);
        CATCH_CHECK(variant_end(variant) <= next_start);
        // The variant is normalized within the previous normalized variant and the next input variant.
        const std::string_view window =
                std::string_view{test_case.ref_seq}.substr(prev_end, next_start - prev_end);
        CATCH_CHECK(
                normalize_variant_vs_ref_core(variant, window, prev_end, test_case.allow_ambig_ref)
                        .first == variant);
        prev_end = variant_end(variant);
        ++i;
    }
}

CATCH_TEST_CASE("Normalization passes reference records through unchanged", TEST_GROUP) {
    struct TestCase {
        std::string name;
        Variant variant;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{"Empty record at a valid position", Variant{0, 0, "", {}, {}, {}, 0.0f, {}, 0, 0}},
        TestCase{"No alternate alleles",
                 Variant{0, 2, "ac", {}, {}, {}, 0.0f, {}, 0, 0}},
        TestCase{"Dot allele",
                 Variant{0, 2, "ac", {"."}, {}, {}, 0.0f, {}, 0, 0}},
        TestCase{"Reference block with annotations",
                 Variant{0, 2, "a", {"<*>"}, {}, {{"END", "6"}}, 20.0f,
                         {{"LEN", "4"}, {"GT", "0/0"}}, 2, 6}},
        TestCase{"Non-variant filter",
                 Variant{0, 2, "ac", {"ac"}, Filters::NonVariant, {}, 0.0f, {}, 0, 0}},
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    // Reference records must not require a reference lookup or have their case changed.
    CATCH_CHECK(normalize_variant_vs_ref_core(test_case.variant, "", 0, false).first ==
                test_case.variant);
    const std::vector<Variant> records{test_case.variant};
    CATCH_CHECK(normalize_variants_vs_ref(records, "", false) == records);
}

CATCH_TEST_CASE("normalize_variant_vs_ref_core returns an error for invalid inputs", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref_seq;
        int64_t prev_variant_end;
        int64_t next_variant_start;
        Variant variant;
        NormalizationStatus expected_status;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{"Negative position", "ACGT", 0, 4,
                 Variant{0, -1, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Position beyond chromosome", "ACGT", 0, 4,
                 Variant{0, 5, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Nonempty REF at chromosome end", "ACGT", 0, 4,
                 Variant{0, 4, "T", {"C"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"REF extends beyond chromosome", "AAAA", 0, 4,
                 Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Empty chromosome with nonempty REF", "", 0, 0,
                 Variant{0, 0, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Unpadded insertion beyond chromosome end", "ACGT", 0, 4,
                 Variant{0, 5, "", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Unpadded insertion at chromosome end", "ACGT", 0, 4,
                 Variant{0, 4, "", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"Empty REF inside the chromosome", "AAAAAA", 0, 6,
                 Variant{0, 3, "", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"Empty REF at chromosome start", "ACGT", 0, 4,
                 Variant{0, 0, "", {"TT"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"REF mismatches the chromosome", "ACGT", 0, 4,
                 Variant{0, 1, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"Variant starts before the window", "AAAA", 3, 4,
                 Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
        TestCase{"ALT equals REF", "ACGT", 0, 4,
                 Variant{0, 2, "G", {"G"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_DUPLICATE_ALLELE},
        TestCase{"REF matching is case-sensitive", "ACGT", 0, 4,
                 Variant{0, 2, "g", {"T"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"One of several ALTs equals REF", "AAAA", 0, 4,
                 Variant{0, 2, "AA", {"A", "AA"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_DUPLICATE_ALLELE},
        TestCase{"All alleles are empty", "ACGT", 0, 4,
                 Variant{0, 2, "", {""}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"Empty REF and one empty ALT", "ACGT", 0, 4,
                 Variant{0, 2, "", {"A", ""}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_REF},
        TestCase{"Insertion on an empty chromosome", "", 0, 0,
                 Variant{0, 0, "", {"A"}, {}, {}, 0.0f, {}, 0, 0}, NormalizationStatus::ERROR_OUT_OF_RANGE},
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    const std::string_view window = std::string_view{test_case.ref_seq}.substr(
            test_case.prev_variant_end, test_case.next_variant_start - test_case.prev_variant_end);
    // Invalid input is returned unchanged with an error status.
    const auto [result_var, result_status] = normalize_variant_vs_ref_core(
            test_case.variant, window, test_case.prev_variant_end, false);
    CATCH_CHECK(result_status == test_case.expected_status);
    CATCH_CHECK(result_var == test_case.variant);
}

CATCH_TEST_CASE("normalize_variants_vs_ref keeps overlapping records unchanged", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref_seq;
        std::vector<Variant> variants;
        std::vector<Variant> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Overlapping SNP records", "AAAA",
            {Variant{0, 1, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 1, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "Overlapping variants are not normalized, even if the first one could shift away", "AAAAAA",
            {Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "Overlapping variants are not normalized, even if trimming the first one would remove the overlap", "CAAAAAAA",
            {Variant{0, 0, "CAA", {"CTA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "A", {""}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "CAA", {"CTA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "A", {""}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "Variant overlaps a reference block", "AAAAAA",
            {Variant{0, 0, "A", {"<*>"}, {}, {{"END", "4"}}, 0.0f, {}, 0, 0},
             Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 0, "A", {"<*>"}, {}, {{"END", "4"}}, 0.0f, {}, 0, 0},
             Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "Later variant is bounded by the furthest end of the previous ones, not only by the last one",
            "ACTTTTTTACG",
            {Variant{0, 1, "CTT", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "T", {"G"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "TT", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 1, "CTT", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "T", {"G"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 4, "TT", {"T"}, {}, {}, 0.0f, {}, 0, 0}},
        },
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    CATCH_CHECK(normalize_variants_vs_ref(test_case.variants, test_case.ref_seq, false) ==
                test_case.expected);
}

CATCH_TEST_CASE("normalize_variant_vs_ref_core returns an error for a negative window start offset",
                TEST_GROUP) {
    // Inputs outside of the window are covered by "normalize_variant_vs_ref_core returns an error for invalid inputs".
    const Variant snp{0, 1, "C", {"T"}, {}, {}, 0.0f, {}, 0, 0};
    const auto [result_var, result_status] = normalize_variant_vs_ref_core(snp, "ACGT", -1, false);
    CATCH_CHECK(result_status == NormalizationStatus::ERROR_NEGATIVE_OFFSET);
    CATCH_CHECK(result_var == snp);
}

CATCH_TEST_CASE("normalize_variants_vs_ref_core with a window start offset", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string chrom;  // Entire chromosome. The window is chrom[win_start_offset:].
        int64_t win_start_offset;
        std::vector<Variant> variants;
        std::vector<Variant> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Already normalized variants are not modified", "ACGTACGT", 2,
            {Variant{0, 3, "T", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 5, "CG", {"C"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 3, "T", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 5, "CG", {"C"}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "First deletion stops at the window start, second at the end of the normalized first",
            "AAAAAAAAAAAA", 3,
            {Variant{0, 6, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 9, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 3, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 5, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
        },
        TestCase{
            "Reference block with an absolute END is kept and blocks left alignment",
            "AAAAAAAAAAAA", 3,
            {Variant{0, 4, "A", {"<*>"}, {}, {{"END", "7"}}, 0.0f, {}, 0, 0},
             Variant{0, 9, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
            {Variant{0, 4, "A", {"<*>"}, {}, {{"END", "7"}}, 0.0f, {}, 0, 0},
             Variant{0, 7, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}},
        },
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    const std::string_view window =
            std::string_view{test_case.chrom}.substr(test_case.win_start_offset);
    const auto [result, result_status] = normalize_variants_vs_ref_core(
            test_case.variants, window, test_case.win_start_offset, false);
    CATCH_CHECK(result == test_case.expected);
    const NormalizationStatus expected_status = (result == test_case.variants)
                                                        ? NormalizationStatus::UNCHANGED
                                                        : NormalizationStatus::CHANGED;
    CATCH_CHECK(result_status == expected_status);
}

CATCH_TEST_CASE(
        "normalize_variants_vs_ref_core returns an error for variants before the window start",
        TEST_GROUP) {
    const std::string_view window{"TACG"};  // Chromosome positions [4, 8).
    const std::vector<Variant> variants{Variant{0, 3, "G", {"C"}, {}, {}, 0.0f, {}, 0, 0}};
    const auto [result, result_status] = normalize_variants_vs_ref_core(variants, window, 4, false);
    CATCH_CHECK(result_status == NormalizationStatus::ERROR_NOT_SORTED);
    CATCH_CHECK(result == variants);
}

CATCH_TEST_CASE("normalize_variants_vs_ref_core returns an error for invalid inputs", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::string ref_seq;
        std::vector<Variant> variants;
        NormalizationStatus expected_status;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Invalid position after a valid variant", "AAAA",
            {Variant{0, 0, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 6, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, NormalizationStatus::ERROR_OUT_OF_RANGE,
        },
        TestCase{
            "Unsorted inputs even when normalization would restore order", "AAAAAA",
            {Variant{0, 4, "A", {"AA"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{0, 2, "A", {"T"}, {}, {}, 0.0f, {}, 0, 0}}, NormalizationStatus::ERROR_NOT_SORTED,
        },
        TestCase{
            "Records on different chromosomes", "AAAA",
            {Variant{0, 0, "A", {"C"}, {}, {}, 0.0f, {}, 0, 0},
             Variant{1, 2, "AA", {"A"}, {}, {}, 0.0f, {}, 0, 0}}, NormalizationStatus::ERROR_NOT_SORTED,
        },
        TestCase{"Negative first position is rejected before normalization", "ACGT", {Variant{}}, NormalizationStatus::ERROR_NOT_SORTED},
    }));
    // clang-format on

    CATCH_INFO("Test name: " << test_case.name);
    // Invalid input is returned unchanged with an error status.
    const auto [result, result_status] =
            normalize_variants_vs_ref_core(test_case.variants, test_case.ref_seq, 0, false);
    CATCH_CHECK(result_status == test_case.expected_status);
    CATCH_CHECK(result == test_case.variants);
    CATCH_CHECK(normalize_variants_vs_ref(test_case.variants, test_case.ref_seq, false) ==
                test_case.variants);
}

}  // namespace dorado::secondary::tests
