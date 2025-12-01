#include "secondary/common/variant.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_base.h"
#include "torch_utils/tensor_utils.h"
#include "utils/ssize.h"

#include <ATen/ATen.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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
        lookup[static_cast<int32_t>(symbols[i])] = i;
    }

    const float fp_prob = (1.0 - tp_prob) / (dorado::ssize(symbols) - 1);

    at::Tensor probs = at::full({dorado::ssize(seq), dorado::ssize(symbols)}, fp_prob,
                                at::TensorOptions().dtype(at::kFloat).device(at::kCPU));

    for (int64_t row = 0; row < dorado::ssize(seq); ++row) {
        const int64_t col = lookup[static_cast<int32_t>(seq[row])];
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

    assert(std::size(cons_seqs) == std::size(true_pos_probs));

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
    all_probs.reserve(cons_seqs.size());
    for (size_t i = 0; i < std::size(cons_seqs); ++i) {
        all_probs.emplace_back(make_haploid_probs(symbols, cons_seqs[i], true_pos_probs[i]));
    }

    return at::stack(all_probs, /*dim=*/1);
}
}  // namespace

namespace dorado::secondary::tests {
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
            "No variants, three haplotype.",
            "ACTG", {"ACTG", "ACTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false, {}, false,
        },

        TestCase{
            "One variant, one haplotype. No normalization.",
            "ACTG", {"AGTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "One variant, two haplotype. No normalization. The genotype information and the alleles are always normalized so they are valid, however.",
            "ACTG", {"AGTG", "ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Two variants, two haplotypes. No normalization.",
            "ACATG", {"AGATG", "ACAAG"}, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 1, 2},
                Variant{0, 3, "T", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 3, 4},
            },
            false,
        },

        TestCase{
            "Multi-base SNP variant, two haplotypes. No normalization.",
            "ACTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 1, "CT", {"CA", "GT"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 1, 3},
            },
            false,
        },


        TestCase{
            "Normalization. One SNP variant, one haplotype. No effect, SNPs cannot be normalized.",
            "AAAA", {"ACAA"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 1, "A", {"C"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Normalization. One deletion variant, one haplotype.",
            "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "CA", {"C"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 0, 3},
            },
            false,
        },

        TestCase{
            "No normalization. One deletion variant, one haplotype. Cannot be represented in the VCF, so puts a '.' in the ALT field.",
            "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 2, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 2, 3},
            },
            false,
        },

        TestCase{
            "Normalization. One deletion variant at position 0, one haplotype. Deletion is the first event, cannot left-extend. Extend to the right with one reference base instead.",
            "ATAC", {"*TAC"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "AT", {"T"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 0, 2},
            },
            false,
        },

        TestCase{
            "Normalization. Single reference base which is deleted in the alt. Cannot be represented in the VCF, so puts a '.' in the ALT field.",
            "A", {"*"}, {0}, {0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 0, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 0, 1},
            },
            false,
        },

        TestCase{
            "Return all reference positions (gVCF). This includes reference positions on variant sites as well.",
            "ACTGA", {"ACAGA"}, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, 3.0f, false, true, true, false, false,
            {
                Variant{0, 0, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 0, 1},
                Variant{0, 1, "C", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 1, 2},
                Variant{0, 2, "T", {"A"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 2, 3},
                Variant{0, 2, "T", {"."}, ".", {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 2, 3},
                Variant{0, 3, "G", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 3, 4},
                Variant{0, 4, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 4, 5},
            },
            false,
        },

        TestCase{
            "Prepending with a reference base because the entire variant is in an insertion.",
            "ACT***GCT", {"ACTAAAGCT"}, {0, 1, 2, 2, 2, 2, 3, 4, 5}, {0, 0, 0, 1, 2, 3, 0, 0, 0}, 3.0f, false, false, true, false, false,
            {
                Variant{0, 2, "T", {"TAAA"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 2, 6},
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
                Variant{0, 6, "A", {"TC"}, "PASS", {}, 70.0f, {{"GT", "1"}, {"GQ", "70"}}, 4, 18},
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
                Variant{0, 6, "TA", {"ATA", "TT"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 4, 20},
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
                Variant{0, 1, "N", {"C", "G"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Ambiguous reference - variants not allowed. 2-base SNP extends into an N reference region. Only one base should be reported.",
            "ANTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, false, false, false, false, false,
            {
                Variant{0, 2, "T", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 2, 3},
            },
            false,
        },

        TestCase{
            "Ambiguous reference IS allowed. 2-base SNP extends into an N reference region.",
            "ANTG", {"AGTG", "ACAG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, 3.0f, true, false, false, false, false,
            {
                Variant{0, 1, "NT", {"CA", "GT"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 1, 3},
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
                Variant{0, 1, "AGCCATT", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 1, 8},
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
                Variant{0, 0, "AGCCATT", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 0, 7},
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
                Variant{0, 6, "AGCCATT", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 6, 13},
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
                Variant{0, 1, "NNNNNAGCCATT", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 1, 13},
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
                Variant{0, 0, "NNNNNAGCCATT", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 0, 12},
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
                Variant{0, 0, "AAGCCATT", {"A"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 0, 8},
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
                Variant{0, 0, "AGCCATTA", {"A"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 0, 8},
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
                Variant{0, 6, "AGCCATTA", {"A"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 6, 14},
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
                Variant{0, 0, "ANNNNNAGCCATT", {"A"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 0, 13},
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
                Variant{0, 0, "NNNNNAGCCATTA", {"A"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 0, 13},
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
                Variant{0, 0, "AAGCCATT", {"A"}, "LowQual", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 0, 8},
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
                Variant{0, 3, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 3, 4},           // V1
                Variant{0, 5, "AA", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 5, 8},          // V2
                Variant{0, 8, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 9, 10},          // V3
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
                Variant{0, 2, "CA", {"C"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 2, 4},             // V1
                Variant{0, 4, "AAA", {"A", "ACC"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 4, 8},     // V2
                Variant{0, 7, "AA", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 8, 10},            // V3
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
                Variant{0, 4, "AAAA", {"A", "CC"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 2, 10},
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
                Variant{0, 3, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 3, 4},           // V1
                Variant{0, 5, "AA", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 5, 8},          // V2
                Variant{0, 8, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 9, 10},          // V3
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
                Variant{0, 2, "CA", {"C"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 2, 4},             // V1
                Variant{0, 4, "AAA", {"A", "AAC"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 4, 8},     // V2
                Variant{0, 7, "AA", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 8, 10},            // V3
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
                Variant{0, 4, "AAAA", {"A", "AC"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 2, 10},
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
                Variant{0, 3, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 3, 4},       // V1
                Variant{0, 5, "AA", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 5, 8},      // V2
                Variant{0, 8, "A", {"."}, ".", {}, 70.0f, {{"GT", "0"}, {"GQ", "70"}}, 9, 10},      // V3
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
                Variant{0, 2, "CA", {"C"}, "PASS", {}, 70.0f, {{"GT", "1/1"}, {"GQ", "70"}}, 2, 4},     // V1
                Variant{0, 4, "AAA", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 4, 8},    // V2
                Variant{0, 7, "AA", {"A"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 8, 10},    // V3
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
                Variant{0, 2, "CAAA", {"C", "CA"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 2, 10},
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
                Variant{0, 2, "CAAA", {"C", "CA"}, "PASS", {}, 70.0f, {{"GT", "1/2"}, {"GQ", "70"}}, 2, 10},
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
                Variant{0, 0, "TTCCCATT", {"T"}, "PASS", {}, 70.0f, {{"GT", "0/1"}, {"GQ", "70"}}, 0, 18},
            },
            false,
        },

    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    const DecoderBase decoder(LabelSchemeType::HAPLOID);

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
    };

    // Create the ungapped draft sequence.
    std::string draft = test_case.ref_seq_with_gaps;
    draft.erase(std::remove(std::begin(draft), std::end(draft), '*'), std::end(draft));

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(general_decode_variants(
                decoder, vc_sample.seq_id, vc_sample.positions_major, vc_sample.positions_minor,
                vc_sample.logits, draft, test_case.pass_min_qual, test_case.ambig_ref,
                test_case.return_all, test_case.normalize, test_case.merge_overlapping,
                test_case.merge_adjacent));

    } else {
        const std::vector<Variant> result = general_decode_variants(
                decoder, vc_sample.seq_id, vc_sample.positions_major, vc_sample.positions_minor,
                vc_sample.logits, draft, test_case.pass_min_qual, test_case.ambig_ref,
                test_case.return_all, test_case.normalize, test_case.merge_overlapping,
                test_case.merge_adjacent);

        CATCH_CHECK(test_case.expected == result);
    }
}

}  // namespace dorado::secondary::tests
