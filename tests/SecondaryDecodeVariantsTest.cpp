#include "secondary/common/variant.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_base.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define TEST_GROUP "[SecondaryConsensus]"

namespace {
at::Tensor make_haploid_probs(const std::string_view symbols,
                              const std::string_view seq,
                              const float tp_prob) {
    if (std::empty(seq)) {
        return at::empty({0, 0}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    }

    // Create a symbol lookup.
    std::array<int32_t, 256> lookup{};
    lookup.fill(static_cast<int32_t>(std::size(symbols)));
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(symbols)); ++i) {
        lookup[static_cast<uint8_t>(symbols[i])] = i;
    }

    const float fp_prob = (1.0 - tp_prob) / (std::ssize(symbols) - 1);

    at::Tensor probs = at::full({std::ssize(seq), std::ssize(symbols)}, fp_prob,
                                at::TensorOptions().dtype(at::kFloat).device(at::kCPU));

    for (int64_t row = 0; row < std::ssize(seq); ++row) {
        const int64_t col = lookup[static_cast<uint8_t>(seq[row])];
        probs.index_put_({row, col}, tp_prob);
    }

    return probs;
}

at::Tensor make_polyploid_probs(const std::string_view symbols,
                                const std::vector<std::string_view>& cons_seqs,
                                const std::vector<float>& true_pos_probs) {
    if (std::empty(cons_seqs)) {
        return at::empty({0, 0, 0}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    }

    if (std::size(cons_seqs) != std::size(true_pos_probs)) {
        throw std::runtime_error{"Expected one true-position probability per haplotype."};
    }

    const size_t len = std::size(cons_seqs[0]);
    for (const std::string_view seq : cons_seqs) {
        if (std::size(seq) != len) {
            throw std::runtime_error("All input sequences need to be of the same length! len: " +
                                     std::to_string(len) +
                                     ", found: " + std::to_string(std::size(seq)));
        }
    }

    // Fill the probabilities for the input sequences.
    std::vector<at::Tensor> all_probs;
    all_probs.reserve(std::size(cons_seqs));
    for (size_t i = 0; i < std::size(cons_seqs); ++i) {
        all_probs.emplace_back(make_haploid_probs(symbols, cons_seqs[i], true_pos_probs[i]));
    }

    return at::stack(all_probs, /*dim=*/1);
}
}  // namespace

namespace dorado::secondary::tests {

CATCH_TEST_CASE("variant_span_helpers", TEST_GROUP) {
    const Variant end_record{
            0,  10, "A", {"<*>"}, {Filters::NonVariant}, {{"END", "15"}}, 0.0f, {{"LEN", "99"}},
            10, 15};
    const Variant len_record{0,  20, "C", {"<*>"}, {Filters::NonVariant}, {}, 0.0f, {{"LEN", "4"}},
                             20, 24};
    const Variant ordinary_record{0, 30, "ACG", {"T"}, {Filters::Pass}, {}, 0.0f, {}, 30, 33};

    CATCH_CHECK(variant_end(end_record) == 15);
    CATCH_CHECK(variant_end(len_record) == 24);
    CATCH_CHECK(variant_end(ordinary_record) == 33);
    CATCH_CHECK(variant_ends_before_position(len_record, 0, 24));
    CATCH_CHECK(!variant_ends_before_position(len_record, 0, 23));
    CATCH_CHECK(variant_covers_position(len_record, 0, 23));
    CATCH_CHECK(!variant_covers_position(len_record, 0, 24));
    CATCH_CHECK(!variant_covers_position(len_record, 1, 23));
}

CATCH_TEST_CASE("extend_span", TEST_GROUP) {
    const std::string draft = "ACGTACGT";

    struct TestCase {
        std::string test_name;
        std::vector<int64_t> positions_major;
        std::vector<int64_t> positions_minor;
        std::string ref_seq_with_gaps;
        std::vector<std::string> cons_seqs_with_gaps;
        int64_t rstart;
        int64_t rend;
        std::vector<std::string> alleles;
        bool prepended;
        int64_t expected_rstart;
        int64_t expected_rend;
        std::vector<std::string> expected_alleles;
    };

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"basic case, no extension",
                     // ACGTACGT
                     // AGGTACGT
                     // ATGTACGT
                     //  ||
                     {0, 1, 2, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 0, 0, 0, 0},
                     "ACGTACGT",
                     {"AGGTACGT", "ATGTACGT"},
                     1,
                     2,
                     {"C", "G", "T"},
                     false,
                     1,
                     2,
                     {"C", "G", "T"}},
            TestCase{"substitution, left flanking minors; extend to include minors but not major",
                     // A**CGTACGT
                     // A**GGTACGT
                     // A**TGTACGT
                     //  |  |
                     {0, 0, 0, 1, 2, 3, 4, 5, 6, 7},
                     {0, 1, 2, 0, 0, 0, 0, 0, 0, 0},
                     "A**CGTACGT",
                     {"A**GGTACGT", "A**TGTACGT"},
                     3,
                     4,
                     {"C", "G", "T"},
                     false,
                     1,
                     4,
                     {"C", "G", "T"}},
            TestCase{"substitution, right flanking minors; extend to include minors but not major",
                     // AC**GTACGT
                     // AG**GTACGT
                     // AT**GTACGT
                     //  |  |
                     {0, 1, 1, 1, 2, 3, 4, 5, 6, 7},
                     {0, 0, 1, 2, 0, 0, 0, 0, 0, 0},
                     "AC**GTACGT",
                     {"AG**GTACGT", "AT**GTACGT"},
                     1,
                     2,
                     {"C", "G", "T"},
                     false,
                     1,
                     4,
                     {"C", "G", "T"}},
            TestCase{"deletion; extend to include left and right majors",
                     // ACGTACGT
                     // ACG**CGT
                     // ATG**CGT
                     //   |   |
                     {0, 1, 2, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 0, 0, 0, 0},
                     "ACGTACGT",
                     {"ACG**CGT", "ATG**CGT"},
                     3,
                     5,
                     {"TA", "", ""},
                     false,
                     2,
                     6,
                     {"GTAC", "GC", "GC"}},
            TestCase{"deletion with left minor; extend to include left and right majors",
                     // ACG*TACGT
                     // ACG**ACGT
                     // ACG*TACGT
                     //   |   |
                     {0, 1, 2, 2, 3, 4, 5, 6, 7},
                     {0, 0, 0, 1, 0, 0, 0, 0, 0},
                     "ACG*TACGT",
                     {"ACG**ACGT", "ATG*TACGT"},
                     3,
                     5,
                     {"T", "", "T"},
                     false,
                     2,
                     6,
                     {"GTA", "GA", "GTA"}},
            TestCase{"deletion with right minor; extend to include left and right majors",
                     // ACGT*ACGT
                     // ACG**ACGT
                     // ATG**ACGT
                     //   |   |
                     {0, 1, 2, 3, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 1, 0, 0, 0, 0},
                     "ACGT*ACGT",
                     {"ACG**ACGT", "ATG**ACGT"},
                     3,
                     4,
                     {"T", "", ""},
                     false,
                     2,
                     6,
                     {"GTA", "GA", "GA"}},
            TestCase{"insertion, left major will already have been prepended; extend to right "
                     "major but not left",
                     // ACGT*ACGT
                     // ACGTTACGT
                     // ATGTAACGT
                     //    |  |
                     {0, 1, 2, 3, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 1, 0, 0, 0, 0},
                     "ACGT*ACGT",
                     {"ACGTTACGT", "ATGTAACGT"},
                     3,
                     5,
                     {"T", "TT", "TA"},
                     true,
                     3,
                     6,
                     {"TA", "TTA", "TAA"}},
            TestCase{"insertion with right minors, left major will already have been prepended; "
                     "extend to right major but not left",
                     // ACGT***ACGT
                     // ACGTT**ACGT
                     // ACGTAA*ACGT
                     //    |    |
                     {0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0},
                     "ACGT***ACGT",
                     {"ACGTT**ACGT", "ATGTAA*ACGT"},
                     3,
                     6,
                     {"T", "TT", "TAA"},
                     true,
                     3,
                     8,
                     {"TA", "TTA", "TAAA"}},
            TestCase{"mixed indel ending on deletion, left major will already have been prepended; "
                     "extend to right major but not left",
                     // ACGT***ACGT
                     // ACGT*CCC*GT
                     // ATGT***CGGT
                     //    |      |
                     {0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0},
                     "ACGT***ACGT",
                     {"ACGT*CCC*GT", "ATGT***CGGT"},
                     3,
                     9,
                     {"TAC", "TCCC", "TCG"},
                     true,
                     3,
                     10,
                     {"TACG", "TCCCG", "TCGG"}},
            TestCase{"mixed indel ending on substitution, left major will already have been "
                     "prepended; don't extend right or left",
                     // ACGT***ACGT
                     // ACGT*CCCGGT
                     // ATGT***CGGT
                     //    |     |
                     {0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7},
                     {0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0},
                     "ACGT***ACGT",
                     {"ACGT*CCCGGT", "ATGT***CGGT"},
                     3,
                     9,
                     {"TAC", "TCCCG", "TCG"},
                     true,
                     3,
                     9,
                     {"TAC", "TCCCG", "TCG"}},
            TestCase{"mixed indel ending on substitution but with right minors, left major will "
                     "already have been prepended; extend right through minors but not major",
                     // ACGT***AC*GT
                     // ACGT*CCCG*GT
                     // ATGT***CG*GT
                     //    |      |
                     {0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6, 7},
                     {0, 0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0},
                     "ACGT***AC*GT",
                     {"ACGT*CCCG*GT", "ATGT***CG*GT"},
                     3,
                     9,
                     {"TAC", "TCCCG", "TCG"},
                     true,
                     3,
                     10,
                     {"TAC", "TCCCG", "TCG"}},
            TestCase{"mixed indel starting with deletion; extend left but not right",
                     // ACGTA***CGT
                     // ACGT*CCCGGT
                     // ATGT***CGGT
                     //    |     |
                     {0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7},
                     {0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0},
                     "ACGTA***CGT",
                     {"ACGT*CCCGGT", "ATGT***CGGT"},
                     4,
                     9,
                     {"AC", "CCCG", "CG"},
                     false,
                     3,
                     9,
                     {"TAC", "TCCCG", "TCG"}},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    std::vector<secondary::ConsensusResult> consensus_results;
    consensus_results.reserve(std::size(test_case.cons_seqs_with_gaps));
    for (const auto& seq : test_case.cons_seqs_with_gaps) {
        consensus_results.emplace_back(secondary::ConsensusResult{"", seq, ""});
    }
    int64_t rstart = test_case.rstart;
    int64_t rend = test_case.rend;
    std::vector<std::string> alleles = test_case.alleles;

    secondary::extend_span(rstart, rend, alleles, test_case.prepended, draft,
                           test_case.positions_major, test_case.positions_minor,
                           test_case.ref_seq_with_gaps, consensus_results);
    CATCH_CHECK(rstart == test_case.expected_rstart);
    CATCH_CHECK(rend == test_case.expected_rend);
    CATCH_CHECK(alleles == test_case.expected_alleles);
}

CATCH_TEST_CASE("decode_variants", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        std::string ref_seq_with_gaps;
        std::vector<std::string_view> consensus_seqs;
        std::vector<int64_t> pos_major;
        std::vector<int64_t> pos_minor;
        float pass_min_qual = 3.0f;
        bool ambig_ref = false;
        bool return_all = false;
        bool normalize = false;
        bool merge_overlapping = false;
        bool merge_adjacent = false;
        std::vector<Variant> expected;
        bool expect_throw = false;
    };

    // Variant source FORMAT tag added by the decoder: both inference-called variants and the gVCF
    // reference blocks it emits are tagged "I", because their qualities come from the inference.
    const std::pair<std::string, std::string> source_inference{secondary::FORMAT_TAG_VARIANT_SOURCE,
                                                               secondary::VARIANT_SOURCE_INFERENCE};

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            "", {}, {}, {}, 3.0f, false, false, false, false, false, {}, false,
        },

        TestCase{
            "No variants, one haplotype.",
            "ACTG", {"ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false, {}, false,
        },

        TestCase{
            "Return all reference positions (gVCF), one haplotype emits haploid reference genotype.",
            "AC", {"AC"}, {0, 1}, {0, 0}, 3.0f, false, true, true, false, false,
            {
                Variant{0, 0, "A", {"<*>"}, {Filters::NonVariant}, {{"END", "2"}}, 60.0f, {source_inference, {"GT", "0"}, {"GQ", "60"}, {"LEN", "2"}, {"MIN_DP", "10"}}, 0, 2},
            },
            false,
        },

        TestCase{
            "Return all reference positions (gVCF), diploid non-ACGT reference bases emit missing genotype.",
            "ANCT", {"A*CT", "A*CT"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, true, true, false, false,
            {
                Variant{0, 0, "A", {"<*>"}, {Filters::NonVariant}, {{"END", "1"}}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"LEN", "1"}, {"MIN_DP", "10"}}, 0, 1},
                Variant{0, 1, "N", {"<*>"}, {Filters::NonVariant}, {{"END", "2"}}, -1.0f, {source_inference, {"GT", "./."}, {"GQ", "."}, {"LEN", "1"}, {"MIN_DP", "10"}}, 1, 2},
                Variant{0, 2, "C", {"<*>"}, {Filters::NonVariant}, {{"END", "4"}}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"LEN", "2"}, {"MIN_DP", "10"}}, 2, 4},
            },
            false,
        },

        TestCase{
            "No variants, three haplotype.",
            "ACTG", {"ACTG", "ACTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false, {}, false,
        },

        TestCase{
            "One variant, one haplotype. No normalization.",
            "ACTG", {"AGTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0"}, {"AQ", "0,60"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "One variant, two haplotype. No normalization. The genotype information and the alleles are always normalized so they are valid, however.",
            "ACTG", {"AGTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Two variants, two haplotypes. No normalization.",
            "ACATG", {"AGATG", "ACAAG"}, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 1, 2},
                Variant{0, 3, "T", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 3, 4},
            },
            false,
        },

        TestCase{
            "Multi-base SNP variant, two haplotypes. No normalization.",
            "ACTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "CT", {"CA", "GT"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,100,200,100,0,200"}, {"AQ", "0,60,60"}}, 1, 3},
            },
            false,
        },


        TestCase{
            "Normalization. One SNP variant, one haplotype. No effect, SNPs cannot be normalized.",
            "AAAA", {"ACAA"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 1, "A", {"C"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0"}, {"AQ", "0,60"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Normalization. One deletion variant, one haplotype.",
            "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "CA", {"C"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0"}, {"AQ", "0,60"}}, 0, 3},
            },
            false,
        },

        TestCase{
            "No normalization. One deletion variant, one haplotype. Cannot be represented in the VCF, so puts a '.' in the ALT field.",
            "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 2, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 2, 3},
            },
            false,
        },

        TestCase{
            "Normalization. One deletion variant at position 0, one haplotype. Deletion is the first event, cannot left-extend. Extend to the right with one reference base instead.",
            "ATAC", {"*TAC"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "AT", {"T"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0"}, {"AQ", "0,60"}}, 0, 2},
            },
            false,
        },

        TestCase{
            "Normalization. Single reference base which is deleted in the alt. Cannot be represented in the VCF, so puts a '.' in the ALT field.",
            "A", {"*"}, {0}, {0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 0, 1},
            },
            false,
        },

        TestCase{
            "Return all reference positions (gVCF), skipping positions covered by variants.",
            "ACTGA", {"ACAGA"}, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, 3.0f, false, true, true, false, false,
            {
                Variant{0, 0, "A", {"<*>"}, {Filters::NonVariant}, {{"END", "2"}}, 60.0f, {source_inference, {"GT", "0"}, {"GQ", "60"}, {"LEN", "2"}, {"MIN_DP", "10"}}, 0, 2},
                Variant{0, 2, "T", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0"}, {"AQ", "0,60"}}, 2, 3},
                Variant{0, 3, "G", {"<*>"}, {Filters::NonVariant}, {{"END", "5"}}, 60.0f, {source_inference, {"GT", "0"}, {"GQ", "60"}, {"LEN", "2"}, {"MIN_DP", "10"}}, 3, 5},
            },
            false,
        },

        TestCase{
            "Prepending with a reference base because the entire variant is in an insertion.",
            "ACT***GCT", {"ACTAAAGCT"}, {0, 1, 2, 2, 2, 2, 3, 4, 5}, {0, 0, 0, 1, 2, 3, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 2, "T", {"TAAA"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "300,0"}, {"AQ", "0,60"}}, 2, 6},
            },
            false,
        },

        TestCase{
            "Edge case. SNPs follow a long stretch of minor positions, making a large variant region. The rstart moved left 1bp because a base was prepended.",
             "CCTAG***********TATTATT",
            {"CCTAG*********TT*CTTATT"
            },
            {0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11},
            {0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 6, "A", {"TC"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "195,0"}, {"AQ", "0,60"}}, 4, 18},
            },
            false,
        },

        TestCase{
            "Edge case, diploid. Normalize, merge overlapping and merge adjacent. SNPs follow a long stretch of minor positions, making a large variant region. The rstart moved left 1bp because a base was prepended.",
             "CCTAG************TTATTATT",
            {"CCTAG*********TT**T*TTATT",
             "CCTAG*********T*AT*ATTATT",
            },
            {0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            {0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, true, true,
            {
                Variant{0, 6, "TA", {"ATA", "TT"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,100,200,100,0,197"}, {"AQ", "0,60,60"}}, 4, 20},
            },
            false,
        },

        TestCase{
            "Ambiguous reference not allowed. Single base SNP, not reported because the reference has an N base.",
            "ANTG", {"AGTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
            },
            false,
        },

        TestCase{
            "Ambiguous reference IS allowed. Single base SNP is reported even though the reference has an N base.",
            "ANTG", {"AGTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, true, false, false, false, false,
            {
                Variant{0, 1, "N", {"C", "G"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,100,100,100,0,100"}, {"AQ", "0,60,60"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Ambiguous reference - variants not allowed. 2-base SNP extends into an N reference region. Only one base should be reported.",
            "ANTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 2, "T", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 2, 3},
            },
            false,
        },

        TestCase{
            "Ambiguous reference IS allowed. 2-base SNP extends into an N reference region.",
            "ANTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, true, false, false, false, false,
            {
                Variant{0, 1, "NT", {"CA", "GT"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "300,100,200,200,0,200"}, {"AQ", "0,60,60"}}, 1, 3},
            },
            false,
        },

        ///////////////////////////////////////
        /// Prepending/appending a ref base ///
        /// if any alt is empty.            ///
        ///////////////////////////////////////
        // No normalization - no prepending or appending.
        TestCase{
            "No normalization. Cannot prepend a base before a large deletion.",
             "AAGCCATTACA",
            {"A*******ACA",
             "A*******ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "AGCCATT", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 1, 8},
            },
            false,
        },

        TestCase{
            "No normalization. Cannot append a base after a large deletion.",
             "AGCCATTACA",
            {"*******ACA",
             "*******ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 0, "AGCCATT", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 0, 7},
            },
            false,
        },

        TestCase{
            "No normalization, so cannot prepend/append. Ambiguous reference NOT allowed. (Appending case.) N bases flank a large deletion. Cannot prepend a ref base because it is an N and ambig_ref == false.",
             "ANNNNNAGCCATTACA",
            {"A************ACA",
             "A************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 6, "AGCCATT", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 6, 13},
            },
            false,
        },

        TestCase{
            "No normalization, so cannot prepend/append. Ambiguous reference IS allowed. With normalization this would be a prepending case (N bases flank a large deletion, but are also themselves deleted; prepends the ref base at position 0).",
             "ANNNNNAGCCATTACA",
            {"A************ACA",
             "A************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, true, false, false, false, false,
            {
                Variant{0, 1, "NNNNNAGCCATT", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 1, 13},
            },
            false,
        },

        TestCase{
            "No normalization, so cannot prepend/append. Ambiguous reference IS allowed. With normalization this would be an appending case (N bases are also deleted but at the very front of the region; this would append the base because there is nothing to prepend).",
             "NNNNNAGCCATTACA",
            {"************ACA",
             "************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, true, false, false, false, false,
            {
                Variant{0, 0, "NNNNNAGCCATT", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 0, 12},
            },
            false,
        },

        // With normalization, prepending/appending should work.
        TestCase{
            "Prepend a base before a large deletion.",
             "AAGCCATTACA",
            {"A*******ACA",
             "A*******ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "AAGCCATT", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 0, 8},
            },
            false,
        },

        TestCase{
            "Append a base after a large deletion.",
             "AGCCATTACA",
            {"*******ACA",
             "*******ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "AGCCATTA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 0, 8},
            },
            false,
        },

        TestCase{
            "Ambiguous reference NOT allowed. (Appending case.) N bases flank a large deletion. Cannot prepend a ref base because it is an N and ambig_ref == false.",
             "ANNNNNAGCCATTACA",
            {"A************ACA",
             "A************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 6, "AGCCATTA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 6, 14},
            },
            false,
        },

        TestCase{
            "Ambiguous reference IS allowed. (Prepending case.) N bases flank a large deletion, but are also themselves deleted. Prepends the ref base at position 0.",
            // Appends the base again because the Ns are also variants (not allowed) even though ambig_ref is true.",
             "ANNNNNAGCCATTACA",
            {"A************ACA",
             "A************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, true, false, true, false, false,
            {
                Variant{0, 0, "ANNNNNAGCCATT", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 0, 13},
            },
            false,
        },

        TestCase{
            "Ambiguous reference IS allowed. (Appending case.) N bases are also deleted but at the very front of the region. This appends the base because there is nothing to prepend.",
             "NNNNNAGCCATTACA",
            {"************ACA",
             "************ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, true, false, true, false, false,
            {
                Variant{0, 0, "NNNNNAGCCATTA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 0, 13},
            },
            false,
        },

        TestCase{
            "Test the PASS min qual filter.",
             "AAGCCATTACA",
            {"A*******ACA",
             "A*******ACA"},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            80.0f, false, false, true, false, false,
            {
                Variant{0, 0, "AAGCCATT", {"A"}, {Filters::LowQual}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "990,697,0"}, {"AQ", "0,60"}}, 0, 8},
            },
            false,
        },

        ///////////////////////////////////////
        ///////////////////////////////////////

        ///// Diploid test 1. /////
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
        //
        // Explanation for test 3 (merge_overlapping and merge_adjacent test):
        //   1. Normalized individual variants are located in positions (check the previous test case: Diploid 1, test 2):
        //       V1: rstart = 2, rend = 4
        //       V2: rstart = 4, rend = 8
        //       V3: rstart = 8, rend = 10
        //    2. These variants are neighboring, and the total region is: rstart = 2, rend = 10.
        //    3. The slice of the pileup for this region is:
        //       - IDX  : 23456789
        //       - MAJOR: 23455678
        //       - MINOR: 00001000
        //       - REF  : CAAA*AAA
        //       - HAP 0: C*A***AA
        //       - HAP 1: C*ACC*A*
        //       - VAR  : 01011101
        //    4. Sequences without deletions:
        //           R : CAAAAAA
        //           H1: CAAA
        //           H2: CACCA
        //    5. Normalization steps:
        //               Input           Right trim      Left trim
        //       rstart: 2               2               3           4
        //           R : CAAAAAA         CAAAAA          AAAAA       AAAA
        //           H1: CAAA        =>  CAA         =>  AA      =>  A       Stop.
        //           H2: CACCA           CACC            ACC         CC
        TestCase{
            "Diploid 1, test 1. No normalization and no filtering of overlapping variants. Report all variants but put dots because empty ALT fields are not allowed in the VCF.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*ACC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 3, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 3, 4},           // V1
                Variant{0, 5, "AA", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 5, 8},          // V2
                Variant{0, 8, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 9, 10},          // V3
            },
            false,
        },
        TestCase{
            "Diploid 1, test 2. Using normalization but no filtering of overlapping variants.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*ACC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,97,0"}, {"AQ", "0,60"}}, 2, 4},             // V1
                Variant{0, 4, "AAA", {"A", "ACC"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "395,200,200,195,0,195"}, {"AQ", "0,60,60"}}, 4, 8},     // V2
                Variant{0, 7, "AA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 8, 10},            // V3
            },
            false,
        },
        TestCase{
            "Diploid 1, test 3. Filter overlapping variants.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*ACC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, true, true,
            {
                Variant{0, 4, "AAAA", {"A", "CC"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "689,395,197,294,0,195"}, {"AQ", "0,60,60"}}, 2, 10},
            },
            false,
        },

        ///// Diploid test 2. /////
        // Explanation for test 3 (merge_overlapping and merge_adjacent test):
        //   1. Normalized individual variants are located in positions (check the previous test case: Diploid 1, test 2):
        //       V1: rstart = 2, rend = 4
        //       V2: rstart = 4, rend = 8
        //       V3: rstart = 8, rend = 10
        //    2. These variants are neighboring, and the total region is: rstart = 2, rend = 10.
        //    3. The slice of the pileup for this region is:
        //       - IDX  : 23456789
        //       - MAJOR: 23455678
        //       - MINOR: 00001000
        //       - REF  : CAAA*AAA
        //       - HAP 0: C*A***AA
        //       - HAP 1: C*AAC*A*
        //       - VAR  : 01011101
        //    4. Sequences without deletions:
        //           R : CAAAAAA
        //           H1: CAAA
        //           H2: CAACA
        //    5. Normalization steps:
        //               Input           Right trim      Left trim
        //       rstart: 2               2               3           4
        //           R : CAAAAAA         CAAAAA          AAAAA       AAAA
        //           H1: CAAA        =>  CAA         =>  AA      =>  A       Stop. No left trim because H1 would be empty.
        //           H2: CAACA           CAAC            AAC         AC
        TestCase{
            "Diploid 2, test 1. No normalization and no filtering of overlapping variants. Report all variants but put dots because empty ALT fields are not allowed in the VCF.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 3, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 3, 4},           // V1
                Variant{0, 5, "AA", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 5, 8},          // V2
                Variant{0, 8, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 9, 10},          // V3
            },
            false,
        },
        TestCase{
            "Diploid 2, test 2. Using normalization but no filtering of overlapping variants. Similar to Diploid 1 tests, "
            "but position 5 does not have a SNP. That means that there could be more trimming at the front of the variant "
            "during normalization, but normalization stops because it hits the previous variant.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,97,0"}, {"AQ", "0,60"}}, 2, 4},             // V1
                Variant{0, 4, "AAA", {"A", "AAC"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "295,100,200,195,0,195"}, {"AQ", "0,60,60"}}, 4, 8},     // V2
                Variant{0, 7, "AA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 8, 10},            // V3
            },
            false,
        },
        TestCase{
            "Diploid 2, test 3. Merge adjacent variants.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAC*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, true, true,
            {
                Variant{0, 4, "AAAA", {"A", "AC"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "589,295,100,294,0,194"}, {"AQ", "0,60,60"}}, 2, 10},
            },
            false,
        },

        ///// Diploid test 3. /////
        // Explanation for test 2 (normalization):
        // Here is an interesting case for V2:
        // The slice of the pileup for this region is:
        //       - IDX  : 23456789
        //       - MAJOR: 23455678
        //       - MINOR: 00001000
        //       - REF  : CAAA*AAA
        //       - HAP 0: C*A***AA
        //       - HAP 1: C*AAA*A*
        //       - VAR  : 01011101
        //                 V1 V2 V3
        // Normalization steps for V2:
        //               Input               Left extend     Right trim
        //       rstart: 5           5       4               4
        //           R : A*A         AA      AAA             AA              Stop. H1 is empty but cannot left-extend because idx=3 is a variant!
        //           H1: ***     =>       => A           =>                  This should fall back to the last step before the "Right trim".
        //           H2: AA*         AA      AAA             AA              This test case triggers the new heuristic in normalization.
        //
        // Explanation for test 3 (merge_overlapping and merge_adjacent test):
        //   1. Normalized individual variants are located in positions (check the previous test case: Diploid 1, test 2):
        //       V1: rstart = 2, rend = 4
        //       V2: rstart = 4, rend = 8
        //       V3: rstart = 8, rend = 10
        //    2. These variants are neighboring, and the total region is: rstart = 2, rend = 10.
        //    3. The slice of the pileup for this region is:
        //       - IDX  : 23456789
        //       - MAJOR: 23455678
        //       - MINOR: 00001000
        //       - REF  : CAAA*AAA
        //       - HAP 0: C*A***AA
        //       - HAP 1: C*AAA*A*
        //       - VAR  : 01011101
        //    4. Sequences without deletions:
        //           R : CAAAAAA
        //           H1: CAAA
        //           H2: CAAAA
        //    5. Normalization steps:
        //               Input           Right trim      Right trim      Right trim      Right trim
        //       rstart: 2               2               2               2               2
        //           R : CAAAAAA         CAAAAA          CAAAA           CAAAA           CAAA
        //           H1: CAAA        =>  CAA         =>  CA          =>  CA          =>  C           Stop. No left trim because H1 would be empty.
        //           H2: CAAAA           CAAA            CAA             CAA             CAA
        TestCase{
            "Diploid 3, test 1. No normalization and no filtering of overlapping variants. Report all variants, but put dots because empty ALT fields are not allowed in the VCF.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAA*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, false, false, false,
            {
                Variant{0, 3, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 3, 4},       // V1
                Variant{0, 5, "AA", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 5, 8},      // V2
                Variant{0, 8, "A", {"."}, {Filters::NonVariant}, {}, 60.0f, {source_inference, {"GT", "0/0"}, {"GQ", "60"}, {"MIN_DP", "10"}}, 9, 10},      // V3
            },
            false,
        },
        TestCase{
            "Diploid 3, test 2. Using normalization but no filtering of overlapping variants. "
            "Triggers the case where left-extend would step into a previous variant.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAA*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, false, false,
            {
                Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "200,97,0"}, {"AQ", "0,60"}}, 2, 4},     // V1
                Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "195,0,200"}, {"AQ", "60,60"}}, 4, 8},    // V2
                Variant{0, 7, "AA", {"A"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "100,0,100"}, {"AQ", "60,60"}}, 8, 10},    // V3
            },
            false,
        },
        TestCase{
            "Diploid 3, test 3. Merge adjacent variants but not overlapping variants.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAA*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, false, true,
            {
                Variant{0, 2, "CAAA", {"C", "CA"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "489,195,94,288,0,94"}, {"AQ", "0,60,60"}}, 2, 10},
            },
            false,
        },
        TestCase{
            "Diploid 3, test 4. Merge adjacent variants AND overlapping variants.",
             "ACCAAA*AAA",
            {"ACC*A***AA",
             "ACC*AAA*A*",
            },
            {0, 1, 2, 3, 4, 5, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            3.0f, false, false, true, true, true,
            {
                Variant{0, 2, "CAAA", {"C", "CA"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "1/2"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "489,195,94,288,0,94"}, {"AQ", "0,60,60"}}, 2, 10},
            },
            false,
        },

        TestCase{
            "Small edge case for normalization, concretely, for right-extension. Initial variant: TCCCATT -> left ext because H0 is empty: TTCCCATT/T/TTCCCATT in region [0, 18] "
            "-> trim: TTCCCAT//TTCCCAT -> right extension (this is the edge case because rend = 18, but last base in this deletion is trimmed, so just bluntly appending a ref base "
            "to existing ref and alt fields would produce an artifact). The right extension should either not run or should reach the A/A/A column. In this case, we stop it preemptively "
            "and keep the previous version of the variant.",
             "T*****TCCCATT*****AGCAATCACCGCCAATTTCTAATTTCATCAATATTTCTATCACCTCAAAATAA",
            {"T*****************AGCAATCACCGCCAATTTCTAATTTCATCAATATTTCTATCACCTCAAAATAA",
             "T*****TCCCATT*****AGCAATCACCGCCAATTTCTAATTTCATCAATATTTCTATCACCTCAAAATAA"},
            {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60},
            {0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            3.0f, false, false, true, true, true,
            {
                Variant{0, 0, "TTCCCATT", {"T"}, {Filters::Pass}, {}, 60.0f, {source_inference, {"GT", "0/1"}, {"GQ", "60"}, {"VQ", "60.00"}, {"DP", "10"}, {"PL", "671,0,700"}, {"AQ", "60,60"}}, 0, 18},
            },
            false,
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    const LabelSchemeType label_scheme = (std::size(test_case.consensus_seqs) == 1)
                                                 ? LabelSchemeType::HAPLOID
                                                 : LabelSchemeType::DIPLOID;

    const DecoderBase decoder(label_scheme);

    // Sanity check that lengths of all elements of the test are good.
    const size_t expected_len = std::size(test_case.ref_seq_with_gaps);
    const bool check_valid_test =
            (std::size(test_case.pos_major) == expected_len) &&
            (std::size(test_case.pos_minor) == expected_len) &&
            std::all_of(std::cbegin(test_case.consensus_seqs), std::cend(test_case.consensus_seqs),
                        [expected_len](const std::string_view s) {
                            return std::size(s) == expected_len;
                        });

    if (!check_valid_test) {
        throw std::runtime_error{"Test is ill formed! Test name: " + test_case.test_name};
    }

    VariantCallingSample vc_sample{
            0,
            test_case.pos_major,
            test_case.pos_minor,
            make_polyploid_probs(decoder.get_label_scheme_symbols(), test_case.consensus_seqs,
                                 std::vector<float>(std::size(test_case.consensus_seqs), 1.0f)),
            at::full({std::ssize(test_case.pos_major)}, 10, at::kInt),
    };

    // Create the ungapped draft sequence.
    std::string draft = test_case.ref_seq_with_gaps;
    draft.erase(std::remove(std::begin(draft), std::end(draft), '*'), std::end(draft));

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(general_decode_variants(
                decoder, vc_sample, draft, test_case.pass_min_qual, 0.0f, 0.0f, 0.0f,
                test_case.ambig_ref, test_case.return_all, test_case.normalize,
                test_case.merge_overlapping, test_case.merge_adjacent,
                secondary::DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS));

    } else {
        const std::vector<Variant> result = general_decode_variants(
                decoder, vc_sample, draft, test_case.pass_min_qual, 0.0f, 0.0f, 0.0f,
                test_case.ambig_ref, test_case.return_all, test_case.normalize,
                test_case.merge_overlapping, test_case.merge_adjacent,
                secondary::DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);

        CATCH_CHECK(test_case.expected == result);
    }
}

CATCH_TEST_CASE("decode_variants adds AD from basecalls", TEST_GROUP) {
    const DecoderBase decoder(LabelSchemeType::HAPLOID);
    const std::string symbols = decoder.get_label_scheme_symbols();

    // Create a symbol lookup.
    std::array<int32_t, 256> lookup{};
    lookup.fill(static_cast<int32_t>(std::size(symbols)));
    for (int32_t i = 0; i < static_cast<int32_t>(std::ssize(symbols)); ++i) {
        lookup[static_cast<uint8_t>(symbols[i])] = i;
    }

    auto make_sample = [&](at::Tensor basecalls, const std::string_view cons_seq) {
        int64_t n_reads = 0;
        if (basecalls.defined()) {
            n_reads = basecalls.size(1);
        }
        const std::vector<std::string> read_ids = placeholder_read_ids(n_reads);
        return VariantCallingSample{
                0,
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                make_haploid_probs(symbols, cons_seq, 0.99f),
                at::full({4}, n_reads, at::kInt),
                read_ids,
                read_ids,
                std::move(basecalls),
        };
    };

    const auto make_basecalls = [&lookup](const std::vector<std::string_view>& seqs) {
        if (std::empty(seqs)) {
            return at::empty({}, at::kChar);
        }

        at::Tensor basecalls = at::zeros({std::ssize(seqs.front()), std::ssize(seqs)}, at::kChar);
        for (int64_t read_idx = 0; read_idx < std::ssize(seqs); ++read_idx) {
            for (int64_t pos = 0; pos < std::ssize(seqs[read_idx]); ++pos) {
                const char c = seqs[read_idx][pos];
                const int8_t encoded = lookup[static_cast<uint8_t>(c)];
                basecalls.index_put_({pos, read_idx}, encoded);
            }
        }
        return basecalls;
    };

    const auto decode = [&](const VariantCallingSample& sample, const std::string_view ref) {
        return general_decode_variants(decoder, sample, ref, 10.0f, 0.0f, 0.0f, 0.0f, false, false,
                                       false, false, false,
                                       DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);
    };

    CATCH_SECTION("counts matching ref and alt allele") {
        const VariantCallingSample sample =
                make_sample(make_basecalls({"ACTA", "ACTA", "ACTA", "ACAA", "ACAA"}), "ACTA");

        const std::vector<Variant> result = decode(sample, "ACAA");

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(std::cbegin(result)->ref == "A");
        CATCH_CHECK(std::cbegin(result)->alts == std::vector<std::string>{"T"});
        const auto& format = std::cbegin(result)->format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        const auto it_ad = format.find("AD");
        CATCH_CHECK(it_ad != format.cend());
        CATCH_CHECK(it_ad->second == "2,3");
    }

    CATCH_SECTION("partially spanning reads don't contribute") {
        const VariantCallingSample sample =
                make_sample(make_basecalls({"**GA", "AG**", "AGGA", "ACAA", "ACAA"}), "ACAA");

        const std::vector<Variant> result = decode(sample, "AGGA");

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(std::cbegin(result)->ref == "GG");
        CATCH_CHECK(std::cbegin(result)->alts == std::vector<std::string>{"CA"});
        const auto& format = std::cbegin(result)->format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        const auto it_ad = format.find("AD");
        CATCH_CHECK(it_ad != std::cend(format));
        CATCH_CHECK(it_ad->second == "1,2");
    }

    CATCH_SECTION("empty basecalls do not add AD") {
        const std::vector<Variant> result = decode(make_sample({}, "ACTA"), "ACAA");

        CATCH_REQUIRE(std::size(result) == 1);
        const auto& format = result.front().format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        CATCH_CHECK(format.find("AD") == std::cend(format));
    }

    CATCH_SECTION("AD tag docs example 1") {
        /**
         * reference     ...A--C-GA...
         * alt           ...A--T-GA...
         * read 1        ...A--T-GA...
         * read 2        ...AAAT-GA...
         * read 3        ...A--TGGA...
         */
        const VariantCallingSample sample{
                0,
                {0, 0, 0, 1, 1, 2, 3},
                {0, 1, 2, 0, 1, 0, 0},
                make_haploid_probs(symbols, "A**T*GA", 0.99f),
                at::full({7}, 3, at::kInt),
                {"read_1", "read_2", "read_3"},
                {"read_1", "read_2", "read_3"},
                make_basecalls({"A--T-GA", "AAAT-GA", "A--TGGA"}),
        };

        const std::vector<Variant> result = general_decode_variants(
                decoder, sample, "ACGA", 0.0f, 0.0f, 0.0f, 0.0f, false, false, true, true, false,
                DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(std::cbegin(result)->ref == "C");
        CATCH_CHECK(std::cbegin(result)->alts == std::vector<std::string>{"T"});
        const auto& format = std::cbegin(result)->format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        const auto it_ad = format.find("AD");
        CATCH_CHECK(it_ad != std::cend(format));
        CATCH_CHECK(it_ad->second == "0,1");
    }

    CATCH_SECTION("AD tag docs example 2") {
        /**
         * reference     ...AACCGA...
         * alt           ...AA--GA...
         * read 1        ...AA--GA...
         * read 2        ...AA---A...
         * read 3        ...A---GA...
         */
        const VariantCallingSample sample{0,
                                          {0, 1, 2, 3, 4, 5},
                                          {0, 0, 0, 0, 0, 0},
                                          make_haploid_probs(symbols, "AA**GA", 0.99f),
                                          at::full({6}, 3, at::kInt),
                                          {"read_1", "read_2", "read_3"},
                                          {"read_1", "read_2", "read_3"},
                                          make_basecalls({"AA--GA", "AA---A", "A---GA"})};

        const std::vector<Variant> result = general_decode_variants(
                decoder, sample, "AACCGA", 0.0f, 0.0f, 0.0f, 0.0f, false, false, true, true, false,
                DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(std::cbegin(result)->ref == "ACC");
        CATCH_CHECK(std::cbegin(result)->alts == std::vector<std::string>{"A"});
        const auto& format = std::cbegin(result)->format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        const auto it_ad = format.find("AD");
        CATCH_CHECK(it_ad != std::cend(format));
        CATCH_CHECK(it_ad->second == "0,1");
    }

    CATCH_SECTION("counts matching ref and diploid alt alleles") {
        const VariantCallingSample sample{
                0,
                {0, 1, 2, 3, 4, 5},
                {0, 0, 0, 0, 0, 0},
                make_polyploid_probs(symbols, {"AA**GA", "AAT*GA"}, {0.99f, 0.99f}),
                at::full({6}, 5, at::kInt),
                {"read_1", "read_2", "read_3", "read_4", "read_5"},
                {"read_1", "read_2", "read_3", "read_4", "read_5"},
                make_basecalls({"AA--GA", "AAT--A", "AA--GA", "AAT-GA", "AATG-A"})};

        const std::vector<Variant> result = general_decode_variants(
                decoder, sample, "AACCGA", 0.0f, 0.0f, 0.0f, 0.0f, false, false, true, true, false,
                DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(std::cbegin(result)->ref == "ACC");
        CATCH_CHECK(std::cbegin(result)->alts == std::vector<std::string>{"A", "AT"});
        const auto& format = std::cbegin(result)->format;
        CATCH_REQUIRE_FALSE(std::empty(format));
        const auto it_ad = format.find("AD");
        CATCH_CHECK(it_ad != std::cend(format));
        CATCH_CHECK(it_ad->second == "0,2,2");
    }
}

CATCH_TEST_CASE("join_samples does not split off a sample start which cannot be merged",
                TEST_GROUP) {
    /**
     * \brief Mimics a real case where a 1-column sample was produced by join_samples:
     *          - S0 has no variants and is split into [0, 1] and [2].
     *          - S1 starts with a deletion at position 5 (positions 3-4 are not in any sample, e.g.
     *              removed by merge-region trimming), and its last non-variant major is at position 6,
     *              so it would be split into [5] and [6, ...]. The left part [5] cannot be merged with
     *              the previous right part [2], so it was emitted as a lone 1-column sample, in which
     *              the deletion cannot be padded and becomes an empty ALT.
     *          - S2 contains only insertion columns.
     *          - S3 continues the insertion and ends with non-variant positions.
     *          The left part of S1 should instead stay with its right part, so that the deletion at 5
     *          can be padded with the following reference base.
     */
    const DecoderBase decoder(LabelSchemeType::DIPLOID);
    const std::string symbols = decoder.get_label_scheme_symbols();
    const std::string draft{"ACGTCACGTA"};

    const auto make_sample = [&](std::vector<int64_t> positions_major,
                                 std::vector<int64_t> positions_minor, const std::string_view hap0,
                                 const std::string_view hap1) {
        const int64_t num_columns = std::ssize(positions_major);
        return VariantCallingSample{
                0,
                std::move(positions_major),
                std::move(positions_minor),
                make_polyploid_probs(symbols, {hap0, hap1}, {0.99f, 0.99f}),
                at::full({num_columns}, 10, at::kInt),
                {},
                {},
                at::Tensor{},
        };
    };

    std::vector<VariantCallingSample> samples;
    samples.emplace_back(make_sample({0, 1, 2}, {0, 0, 0}, "ACG", "ACG"));
    samples.emplace_back(make_sample({5, 6, 7, 7, 7}, {0, 0, 0, 1, 2}, "*CTTT", "ACTTT"));
    samples.emplace_back(make_sample({7, 7}, {3, 4}, "TT", "TT"));
    samples.emplace_back(make_sample({7, 8, 9}, {5, 0, 0}, "TTA", "TTA"));

    const std::vector<VariantCallingSample> joined =
            join_samples(std::move(samples), draft, decoder);

    // S0 stays whole, and S1 is joined with S2 and the left part of S3.
    std::vector<std::vector<int64_t>> joined_positions_major;
    joined_positions_major.reserve(std::size(joined));
    for (const VariantCallingSample& sample : joined) {
        joined_positions_major.emplace_back(sample.positions_major);
    }
    const std::vector<std::vector<int64_t>> expected_positions_major{
            {0, 1, 2}, {5, 6, 7, 7, 7, 7, 7, 7, 8}, {9}};
    CATCH_CHECK(joined_positions_major == expected_positions_major);

    // The deletion at position 5 is padded with the following reference base.
    std::vector<Variant> result;
    for (const VariantCallingSample& sample : joined) {
        std::vector<Variant> variants = general_decode_variants(
                decoder, sample, draft, 0.0f, 0.0f, 0.0f, 0.0f, false, false, true, true, false,
                DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);
        result.insert(std::end(result), std::begin(variants), std::end(variants));
    }
    CATCH_CHECK(std::none_of(std::begin(result), std::end(result),
                             [](const Variant& var) { return is_reference_record(var); }));
    CATCH_REQUIRE_FALSE(std::empty(result));
    CATCH_CHECK(result.front().pos == 5);
    CATCH_CHECK(result.front().ref == "AC");
    CATCH_CHECK(result.front().alts == std::vector<std::string>{"C"});
}

CATCH_TEST_CASE("join_samples joins a long insertion split between multiple samples", TEST_GROUP) {
    /**
     * \brief An insertion of 10 bases after position 2 is longer than a sample, so it is split between
     *          four samples. The two middle samples contain only insertion columns. All of the samples
     *          must be joined into a single sample, so that the insertion is decoded as one variant.
     */
    const DecoderBase decoder(LabelSchemeType::DIPLOID);
    const std::string symbols = decoder.get_label_scheme_symbols();
    const std::string draft{"ACGTA"};

    const auto make_sample = [&](std::vector<int64_t> positions_major,
                                 std::vector<int64_t> positions_minor, const std::string_view hap) {
        const int64_t num_columns = std::ssize(positions_major);
        return VariantCallingSample{
                0,
                std::move(positions_major),
                std::move(positions_minor),
                make_polyploid_probs(symbols, {hap, hap}, {0.99f, 0.99f}),
                at::full({num_columns}, 10, at::kInt),
                {},
                {},
                at::Tensor{},
        };
    };

    std::vector<VariantCallingSample> samples;
    samples.emplace_back(make_sample({0, 1, 2, 2, 2}, {0, 0, 0, 1, 2}, "ACGAC"));
    samples.emplace_back(make_sample({2, 2, 2}, {3, 4, 5}, "GTT"));
    samples.emplace_back(make_sample({2, 2, 2}, {6, 7, 8}, "GCA"));
    samples.emplace_back(make_sample({2, 2, 3, 4}, {9, 10, 0, 0}, "ACTA"));

    const std::vector<VariantCallingSample> joined =
            join_samples(std::move(samples), draft, decoder);

    // All insertion columns are in the same sample.
    std::vector<std::vector<int64_t>> joined_positions_minor;
    joined_positions_minor.reserve(std::size(joined));
    for (const VariantCallingSample& sample : joined) {
        joined_positions_minor.emplace_back(sample.positions_minor);
    }
    const std::vector<std::vector<int64_t>> expected_positions_minor{
            {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0}, {0}};
    CATCH_CHECK(joined_positions_minor == expected_positions_minor);

    std::vector<Variant> result;
    for (const VariantCallingSample& sample : joined) {
        std::vector<Variant> variants = general_decode_variants(
                decoder, sample, draft, 0.0f, 0.0f, 0.0f, 0.0f, false, false, true, true, false,
                DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);
        result.insert(std::end(result), std::begin(variants), std::end(variants));
    }
    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result.front().pos == 2);
    CATCH_CHECK(result.front().ref == "G");
    CATCH_CHECK(result.front().alts == std::vector<std::string>{"GACGTTGCAAC"});
}

CATCH_TEST_CASE("compute VAF from AD", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        std::vector<std::string> alts;
        std::unordered_map<std::string, std::string> format;
        std::vector<float> expected;
        int32_t decimal_places;
    };

    auto [test_case] = GENERATE_REF(table<TestCase>(
            {TestCase{"calculates and rounds VAF values for all alternate alleles",
                      {"T", "G"},
                      {{"AD", "10,7,3"}, {"DP", "30"}},
                      {0.23f, 0.1f},
                      2},
             TestCase{"rounding truncates to specified decimal places",
                      {"T", "G"},
                      {{"AD", "10,7,3"}, {"DP", "30"}},
                      {0.233f, 0.1f},
                      3},
             TestCase{"prefers MIN_DP over DP",
                      {"T"},
                      {{"AD", "4,6"}, {"DP", "20"}, {"MIN_DP", "10"}},
                      {0.6f},
                      2},
             TestCase{"returns empty for an empty alts", {}, {{"AD", "0"}, {"DP", "10"}}, {}, 2},
             TestCase{"returns empty for a reference record",
                      {"<*>"},
                      {{"AD", "1"}, {"DP", "10"}},
                      {},
                      2},
             TestCase{"returns empty when AD is missing", {"T", "G"}, {{"DP", "30"}}, {}, 2},
             TestCase{"returns empty when depth is missing", {"T", "G"}, {{"AD", "10,7,3"}}, {}, 2},
             TestCase{"returns empty for zero depth",
                      {"T", "G"},
                      {{"AD", "10,7,3"}, {"DP", "0"}},
                      {},
                      2},
             TestCase{"returns empty for negative depth",
                      {"T", "G"},
                      {{"AD", "10,7,3"}, {"DP", "-1"}},
                      {},
                      2},
             TestCase{"returns empty for too many AD values",
                      {"T"},
                      {{"AD", "10,7,3"}, {"DP", "20"}},
                      {},
                      2},
             TestCase{"returns empty for too few AD values",
                      {"T", "G"},
                      {{"AD", "10,7"}, {"DP", "20"}},
                      {},
                      2}}));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    const Variant var{.ref = "A",
                      .alts = test_case.alts,
                      .filters = {Filters::Pass},
                      .info = {},
                      .format = test_case.format};
    const std::vector<float> result = compute_VAF_from_AD(var, test_case.decimal_places);
    CATCH_CHECK(result == test_case.expected);
}

CATCH_TEST_CASE("lookup GT case lists", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        int64_t ploidy;
        int64_t num_alleles;
        std::vector<std::vector<int32_t>> expected_gts;
        bool expect_throw = false;
    };

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"ploidy 1, num_alleles 1", 1, 1, {{0}}, false},
            TestCase{"ploidy 1, num_alleles 2", 1, 2, {{0}, {1}}, false},
            TestCase{"ploidy 1, num_alleles 3", 1, 3, {{0}, {1}, {2}}, false},
            TestCase{"ploidy 2, num_alleles 1", 2, 1, {{0, 0}}, false},
            TestCase{"ploidy 2, num_alleles 2", 2, 2, {{0, 0}, {0, 1}, {1, 1}}, false},
            TestCase{"ploidy 2, num_alleles 3",
                     2,
                     3,
                     {{0, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 2}},
                     false},
            TestCase{"ploidy 2, num_alleles 4",
                     2,
                     4,
                     {{0, 0},
                      {0, 1},
                      {1, 1},
                      {0, 2},
                      {1, 2},
                      {2, 2},
                      {0, 3},
                      {1, 3},
                      {2, 3},
                      {3, 3}},
                     false},
            TestCase{"ploidy 0, invalid", 0, 1, {}, true},
            TestCase{"ploidy out of range", 3, 1, {}, true},
            TestCase{"ploidy 1, num_alleles out of range", 1, 4, {}, true},
            TestCase{"ploidy 2, num_alleles out of range", 2, 5, {}, true},
            TestCase{"num_alleles 0, invalid", 2, 0, {}, true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(lookup_gts(test_case.ploidy, test_case.num_alleles));
    } else {
        const auto [num_elements, gt_list] = lookup_gts(test_case.ploidy, test_case.num_alleles);
        CATCH_CHECK(num_elements == std::ssize(test_case.expected_gts));
        CATCH_CHECK(num_elements == std::ssize(gt_list));
        CATCH_CHECK(std::vector<std::vector<int32_t>>(std::cbegin(gt_list), std::cend(gt_list)) ==
                    test_case.expected_gts);
    }
}

CATCH_TEST_CASE("split GT string", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        std::string gt;
        std::vector<std::string_view> expected_strings;
        bool expected_phased{false};
        bool expect_throw{false};
    };

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"empty", "", {}},
            TestCase{"haploid", "0", {"0"}},
            TestCase{"diploid unphased", "0/1", {"0", "1"}},
            TestCase{"diploid phased", "1|2", {"1", "2"}, true},
            TestCase{"diploid phased, optional first indicator present", "|1|2", {"1", "2"}, true},
            TestCase{"diploid, empty segment dropped", "1//2", {"1", "2"}},
            TestCase{"triploid unphased", "1/2/2", {"1", "2", "2"}},
            TestCase{"triploid mixed phasing throws", "|1|2/2", {}, false, true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(split_GT_string(test_case.gt));
    } else {
        const auto [result_strings, result_phased] = split_GT_string(test_case.gt);
        CATCH_CHECK(result_strings == test_case.expected_strings);
        CATCH_CHECK(result_phased == test_case.expected_phased);
    }
}

}  // namespace dorado::secondary::tests
