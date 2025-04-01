#include "secondary/consensus/variant_calling.h"
#include "secondary/variant.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace dorado::secondary::consensus::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("normalize_variant", TEST_GROUP) {
    spdlog::set_level(spdlog::level::trace);

    const std::string reference_1{"GGGGCATGGGG"};
    const std::vector<std::string_view> consensus_1{"GGGGTGCGGGG"};
    const std::vector<int64_t> pos_major_1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::vector<int64_t> pos_minor_1{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::string reference_2{"GGGCACACACAGGG"};
    const std::vector<std::string_view> consensus_2{"GGGCACACACAGGG"};
    const std::vector<int64_t> pos_major_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const std::vector<int64_t> pos_minor_2{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // clang-format off
    //                                                         V1 V2    -> Variants
    //                                                          v vv
    const std::string reference_3{                  "GCCAACTTACCAAAAAAAAAA"};
    const std::vector<std::string_view> consensus_3{"GCCAACTTACC*A**AAAAAA"};
    const std::vector<int64_t> pos_major_3{0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    const std::vector<int64_t> pos_minor_3{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0};
    // clang-format on

    const std::string reference_4{"ACCAAAAAA"};
    const std::vector<std::string_view> consensus_4{"ACC*A**AA", "ACC*AC*A*"};
    const std::vector<int64_t> pos_major_4{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int64_t> pos_minor_4{0, 0, 0, 0, 0, 0, 0, 0, 0};

    const std::string reference_5{"ACCAAA*AAA"};
    const std::vector<std::string_view> consensus_5{"ACC*A***AA", "ACC*ACC*A*"};
    const std::vector<int64_t> pos_major_5{0, 1, 2, 3, 4, 5, 5, 6, 7, 8};
    const std::vector<int64_t> pos_minor_5{0, 0, 0, 0, 0, 0, 1, 0, 0, 0};

    struct TestCase {
        std::string test_name;
        std::string ref_seq_with_gaps;
        std::vector<std::string_view> consensus_seqs;
        std::vector<int64_t> pos_major;
        std::vector<int64_t> pos_minor;
        Variant variant;
        Variant expected;
        bool expect_throw = false;
    };

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            "", {}, {}, {}, {}, {}, false,
        },
        TestCase{
            "Haploid, parsimony, Left trim",
            reference_1, consensus_1, pos_major_1, pos_minor_1,
            Variant{0, 3, "GCAT", {"GTGC"}, {}, {}, 30.0f, {}, 3, 7},
            Variant{0, 4, "CAT", {"TGC"}, {}, {}, 30.0f, {}, 3, 7},
            false,
        },
        TestCase{
            "Haploid, parsimony, right trim",
            reference_1, consensus_1, pos_major_1, pos_minor_1,
            Variant{0, 4, "CATG", {"TGCG"}, {}, {}, 31.0f, {}, 4, 8},
            Variant{0, 4, "CAT", {"TGC"}, {}, {}, 31.0f, {}, 4, 8},
            false,
        },
        TestCase{
            "Haploid, parsimony, left and right trim",
            reference_1, consensus_1, pos_major_1, pos_minor_1,
            Variant{0, 3, "GCATG", {"GTGCG"}, {}, {}, 32.0f, {}, 3, 8},
            Variant{0, 4, "CAT", {"TGC"}, {}, {}, 32.0f, {}, 3, 8},
            false,
        },
        TestCase{
            "Haploid, parsimony, already trimmed",
            reference_1, consensus_1, pos_major_1, pos_minor_1,
            Variant{0, 4, "CAT", {"TGC"}, {}, {}, 33.0f, {}, 4, 7},
            Variant{0, 4, "CAT", {"TGC"}, {}, {}, 33.0f, {}, 4, 7},
            false,
        },

        /// Medaka normalization tests, ported from test_vcf.py.
        TestCase{
            "Haploid, normalize, left align and empty alt",
            reference_2, consensus_2, pos_major_2, pos_minor_2,
            Variant{0, 7, "CA", {""}, {}, {}, 34.0f, {}, 7, 9},
            Variant{0, 2, "GCA", {"G"}, {}, {}, 34.0f, {}, 2, 9},
            false,
        },
        TestCase{
            "Haploid, normalize, left align and alt has a redundant base",
            reference_2, consensus_2, pos_major_2, pos_minor_2,
            Variant{0, 5, "CAC", {"C"}, {}, {}, 35.0f, {}, 5, 8},
            Variant{0, 2, "GCA", {"G"}, {}, {}, 35.0f, {}, 2, 8},
            false,
        },
        TestCase{
            "Haploid, normalize, right trim",
            reference_2, consensus_2, pos_major_2, pos_minor_2,
            Variant{0, 2, "GCACA", {"GCA"}, {}, {}, 36.0f, {}, 2, 7},
            Variant{0, 2, "GCA", {"G"}, {}, {}, 36.0f, {}, 2, 7},
            false,
        },
        TestCase{
            "Haploid, normalize, left trim",
            reference_2, consensus_2, pos_major_2, pos_minor_2,
            Variant{0, 1, "GGCA", {"GG"}, {}, {}, 37.0f, {}, 1, 5},
            Variant{0, 2, "GCA", {"G"}, {}, {}, 37.0f, {}, 1, 5},
            false,
        },

        /// New tests for complicated normalization with overlapping variants.
        TestCase{
            "Real test case, non-adjacent deletions in a homopolymer, variant V1. Should be normalized fully until a non A-base is reached.",
            reference_3, consensus_3, pos_major_3, pos_minor_3,
            Variant{0, 11, "A", {""}, {}, {}, 38.0f, {}, 11, 12},
            Variant{0, 10, "CA", {"C"}, {}, {}, 38.0f, {}, 10, 12},
            false,
        },
        TestCase{
            "Real test case, non-adjacent deletions in a homopolymer, variant V2. Stops normalization when it hits original V1 coordinates.",
            reference_3, consensus_3, pos_major_3, pos_minor_3,
            Variant{0, 13, "AA", {""}, {}, {}, 39.0f, {}, 13, 15},
            Variant{0, 12, "AAA", {"A"}, {}, {}, 39.0f, {}, 12, 15},
            false,
        },

        TestCase{
            "Haploid, edge case",
             "GCT*TTGTGGGCTGGA",
            {"GCTCGGGTGGGCTGGA"},
            {0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0},
            Variant{0, 2, "TT", {"CGG"}, {}, {}, 50.0f, {}, 3, 6},
            Variant{0, 3, "TT", {"CGG"}, {}, {}, 50.0f, {}, 3, 6},
            false,
        },

        TestCase{
            "Haploid, variant near the beginning of the window, but this is not the first window (so 'rstart' is smaller than 'pos').",
             "TGATAT",
            {"T*****"},
            {1270331, 1270332, 1270333, 1270334, 1270335, 1270336},
            {0, 0, 0, 0, 0, 0},
            Variant{0, 1270332, "GATAT", {""}, {}, {}, 51.0f, {}, 1, 6},
            Variant{0, 1270331, "TGATAT", {"T"}, {}, {}, 51.0f, {}, 0, 6},
            false,
        },

        // Diploid case with 3 overlapping variants.
        //           <same>
        //             012 345678
        //     REF:    ACC|AAAAAA
        //     HAP1:   ACC|*A**AA
        //     HAP2:   ACC|*AC*A*
        //                 ^ ^^ ^
        //               V1 V2 V3
        //     Three variants in this region:
        //         - V1: 'A'  -> ('', ''), pos = 3 - 4
        //         - V2: 'AA' -> ('', 'C'), pos = 5 - 7
        //         - V3: 'A'  -> ('A', ''), pos = 8 - 9
        //
        //     If V3 gets normalized to the left, it will run over V2.
        //     The legacy normalization procedure does not take into account the haplotype
        //     sequence, so it would normalize over a SNP adjacent to an insertion, and
        //     one insertion here would get lost.
        //
        //     In this new version of normalization, we consider the actual haplotype base
        //     and prevent running over a variant bluntly.
        TestCase{
            "Diploid 1, variant V1, normalize with an overlapping variant",
            reference_4, consensus_4, pos_major_4, pos_minor_4,
            Variant{0, 3, "A", {"", ""}, {}, {}, 40.0f, {}, 3, 4},
            Variant{0, 2, "CA", {"C", "C"}, {}, {}, 40.0f, {}, 2, 4},
            false,
        },
        TestCase{
            "Diploid 1, variant V2, normalize with an overlapping variant",
            reference_4, consensus_4, pos_major_4, pos_minor_4,
            Variant{0, 5, "AA", {"", "C"}, {}, {}, 41.0f, {}, 5, 7},
            Variant{0, 4, "AAA", {"A", "AC"}, {}, {}, 41.0f, {}, 4, 7},
            false,
        },
        TestCase{
            "Diploid 1, variant V3, normalize with an overlapping variant",
            reference_4, consensus_4, pos_major_4, pos_minor_4,
            Variant{0, 8, "A", {"A", ""}, {}, {}, 42.0f, {}, 8, 9},
            Variant{0, 7, "AA", {"AA", "A"}, {}, {}, 42.0f, {}, 7, 9},
            false,
        },

        // Diploid case with minor reference positions.
        //           <same>
        //             012 3456789 Indices
        //             012 3455678 Major positions
        //             000 0001000 Minor positions
        //     REF:    ACC|AAA*AAA
        //     HAP1:   ACC|*A***AA
        //     HAP1:   ACC|*ACC*A*
        //                 ^ ^^^ ^
        //                 V1 V2 V3
        //     Three variants in this region:
        //         - V1: 'A'  -> ('', ''), pos = 3 - 4
        //         - V2: 'AA' -> ('', 'CC'), pos = 5 - 7, but it spans 3 columns because of
        //                                                 a minor position.
        //         - V3: 'A'  -> ('A', ''), pos = 8 - 9
        TestCase{
            "Diploid 2, variant V1, normalize with an overlapping variant",
            reference_5, consensus_5, pos_major_5, pos_minor_5,
            Variant{0, 3, "A", {"", ""}, {}, {}, 43.0f, {}, 3, 4},
            Variant{0, 2, "CA", {"C", "C"}, {}, {}, 43.0f, {}, 2, 4},
            false,
        },
        TestCase{
            "Diploid 2, variant V2, normalize with an overlapping variant",
            reference_5, consensus_5, pos_major_5, pos_minor_5,
            Variant{0, 5, "AA", {"", "CC"}, {}, {}, 44.0f, {}, 5, 8},
            Variant{0, 4, "AAA", {"A", "ACC"}, {}, {}, 44.0f, {}, 4, 8},
            false,
        },
        TestCase{
            "Diploid 2, variant V3, normalize with an overlapping variant",
            reference_5, consensus_5, pos_major_5, pos_minor_5,
            Variant{0, 8, "A", {"A", ""}, {}, {}, 45.0f, {}, 9, 10},
            Variant{0, 7, "AA", {"AA", "A"}, {}, {}, 45.0f, {}, 8, 10},
            false,
        },

        // Edge case where a SNP follows an indel variant.
        // Normalize the start of the variant. For example, if the input variant represents a region like this:
        // - POS  :      43499195    43499196
        //               v           v
        // - REF  : CCTAG************TTATTATT
        // - HAP 0: CCTAG*********TT**T*TTATT
        // - HAP 1: CCTAG*********T*AT*ATTATT
        // - VAR  : 0000011111111111111100000
        // - MARK :      ^
        //
        // it is possible that the input variant.pos was set to the pos_major of the beginning of the variant
        // (in this case, on a minor position which does not contain a reference base).
        // While actually, the variant.pos should have been set to the first major position after rstart.
        TestCase{
            "SNPs follow a long stretch of minor positions, making a large variant region",
             "CCTAG************TTATTATT",
            {"CCTAG*********TT**T*TTATT",
             "CCTAG*********T*AT*ATTATT",
            },
            {
                43499191, 43499192, 43499193, 43499194, 43499195, 43499195, 43499195, 43499195,
                43499195, 43499195, 43499195, 43499195, 43499195, 43499195, 43499195, 43499195,
                43499195, 43499196, 43499197, 43499198, 43499199, 43499200, 43499201, 43499202,
                43499203
            },
            {0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0},
            Variant{0, 43499195, "TTA", {"TTT", "TATA"}, {}, {}, 3.0f, {}, 5, 25},
            Variant{0, 43499197, "TA", {"TT", "ATA"}, {}, {}, 3.0f, {}, 5, 25},
            false,
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(normalize_variant(test_case.ref_seq_with_gaps, test_case.consensus_seqs,
                                             test_case.pos_major, test_case.pos_minor,
                                             test_case.variant));
    } else {
        const Variant result =
                normalize_variant(test_case.ref_seq_with_gaps, test_case.consensus_seqs,
                                  test_case.pos_major, test_case.pos_minor, test_case.variant);
        CATCH_CHECK(test_case.expected == result);
    }
}

}  // namespace dorado::secondary::consensus::tests