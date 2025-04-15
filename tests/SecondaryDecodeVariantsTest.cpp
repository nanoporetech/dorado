#include "secondary/consensus/variant_calling.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_base.h"
#include "secondary/variant.h"
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

    // Fill the probabilities for the input sequences.
    std::vector<at::Tensor> all_probs;
    for (size_t i = 0; i < std::size(cons_seqs); ++i) {
        all_probs.emplace_back(make_haploid_probs(symbols, cons_seqs[i], true_pos_probs[i]));
    }

    return at::stack(all_probs);
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
        bool normalize = false;
        bool ambig_ref = false;
        bool return_all = false;
        std::vector<Variant> expected;
        bool expect_throw = false;
    };

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            "", {}, {}, {}, false, false, false, {}, false,
        },

        TestCase{
            "No variants, one haplotype.",
            "ACTG", {"ACTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, false, false, false, {}, false,
        },

        TestCase{
            "One variant, one haplotype. No normalization.",
            "ACTG", {"AGTG"}, {0, 1, 2, 3}, {0, 0, 0, 0}, false, false, false,
            {
                Variant{0, 1, "C", {"G"}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Normalization. One SNP variant, one haplotype. No effect, SNPs cannot be normalized.",
            "AAAA", {"ACAA"}, {0, 1, 2, 3}, {0, 0, 0, 0}, true, false, false,
            {
                Variant{0, 1, "A", {"C"}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 1, 2},
            },
            false,
        },

        TestCase{
            "Normalization. One deletion variant, one haplotype.",
            "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, true, false, false,
            {
                Variant{0, 0, "CA", {"C"}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 0, 3},
            },
            false,
        },

        // Enable this test when normalization can be turned off in decode_variants.
        // TestCase{
        //     "No normalization. One deletion variant, one haplotype.",
        //     "CAAA", {"CA*A"}, {0, 1, 2, 3}, {0, 0, 0, 0}, false, false, false,
        //     {
        //         Variant{0, 2, "A", {""}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 2, 3},
        //     },
        //     false,
        // },

        TestCase{
            "Normalization. One deletion variant at position 0, one haplotype. Deletion is the first event, cannot left-extend. Extend to the right with one reference base instead.",
            "ATAC", {"*TAC"}, {0, 1, 2, 3}, {0, 0, 0, 0}, true, false, false,
            {
                Variant{0, 0, "AT", {"T"}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 0, 2},
            },
            false,
        },

        TestCase{
            "Normalization. Single reference base which is deleted in the alt.",
            "A", {"*"}, {0}, {0}, true, false, false,
            {
                Variant{0, 0, "A", {""}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 0, 1},
            },
            false,
        },

        TestCase{
            "Return all reference positions (gVCF). This includes reference positions on variant sites as well.",
            "ACTGA", {"ACAGA"}, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, true, false, true,
            {
                Variant{0, 0, "A", {"."}, ".", {}, 70.0f, {{"GT", 0}, {"GQ", 70}}, 0, 1},
                Variant{0, 1, "C", {"."}, ".", {}, 70.0f, {{"GT", 0}, {"GQ", 70}}, 1, 2},
                Variant{0, 2, "T", {"A"}, "PASS", {}, 70.0f, {{"GT", 1}, {"GQ", 70}}, 2, 3},
                Variant{0, 2, "T", {"."}, ".", {}, 0.0f, {{"GT", 0}, {"GQ", 0}}, 2, 3},
                Variant{0, 3, "G", {"."}, ".", {}, 70.0f, {{"GT", 0}, {"GQ", 70}}, 3, 4},
                Variant{0, 4, "A", {"."}, ".", {}, 70.0f, {{"GT", 0}, {"GQ", 70}}, 4, 5},
            },
            false,
        },

        TestCase{
            "Prepending with a reference base because the entire variant is in an insertion.",
            "ACT***GCT", {"ACTAAAGCT"}, {0, 1, 2, 2, 2, 2, 3, 4, 5}, {0, 0, 0, 1, 2, 3, 0, 0, 0}, true, false, false,
            {
                Variant{0, 2, "T", {"TAAA"}, "PASS", {}, 210.0f, {{"GT", 1}, {"GQ", 210}}, 2, 6},
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
            true, false, false,
            {
                Variant{0, 6, "A", {"TC"}, "PASS", {}, 280.0f, {{"GT", 1}, {"GQ", 280}}, 4, 18},
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

    // Workaround - for now, decode_variants is for haploid inputs only. Take probs only for the first haplotype.
    if (!std::empty(test_case.consensus_seqs)) {
        vc_sample.logits = vc_sample.logits.index({0});
    }

    // Create the ungapped draft sequence.
    std::string draft = test_case.ref_seq_with_gaps;
    draft.erase(std::remove(std::begin(draft), std::end(draft), '*'), std::end(draft));

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(decode_variants(decoder, vc_sample, draft, test_case.ambig_ref,
                                           test_case.return_all));

    } else {
        const std::vector<Variant> result = decode_variants(
                decoder, vc_sample, draft, test_case.ambig_ref, test_case.return_all);

        CATCH_CHECK(test_case.expected == result);
    }
}

}  // namespace dorado::secondary::tests
