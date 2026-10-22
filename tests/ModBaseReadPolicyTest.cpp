#include "read_pipeline/base/messages/DuplexRead.h"
#include "read_pipeline/base/messages/ReadCommon.h"
#include "read_pipeline/nodes/ModBaseReadPolicies.h"
#include "utils/sequence_utils.h"

#include <ATen/ops/arange.h>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#define TEST_GROUP "[modbase_policy]"

namespace {

using namespace dorado;

constexpr size_t STRIDE = 2;
const std::string SEQUENCE = "ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT";

std::vector<uint8_t> one_move_per_base(const size_t num_bases) {
    std::vector<uint8_t> moves;
    for (size_t i = 0; i < num_bases; ++i) {
        moves.push_back(1);
        moves.push_back(0);
    }
    return moves;
}

at::Tensor make_signal(const std::vector<uint8_t>& moves) {
    return at::arange(static_cast<int64_t>(moves.size() * STRIDE), at::kFloat);
}

DuplexRead make_duplex_read() {
    DuplexRead read;
    read.read_common.seq = SEQUENCE;

    auto& inputs = read.stereo_feature_inputs;
    inputs.template_seq = SEQUENCE;
    inputs.template_moves = one_move_per_base(SEQUENCE.size());
    inputs.template_signal = make_signal(inputs.template_moves);

    // The complement strand is stored in the same orientation as the duplex sequence.
    inputs.complement_seq = SEQUENCE;
    inputs.complement_moves = one_move_per_base(SEQUENCE.size());
    inputs.complement_signal = make_signal(inputs.complement_moves);
    return read;
}

}  // namespace

CATCH_TEST_CASE(TEST_GROUP ": simplex policy yields a single strand", TEST_GROUP) {
    ReadCommon read;
    read.seq = SEQUENCE;
    read.moves = one_move_per_base(SEQUENCE.size());
    read.raw_data = make_signal(read.moves);

    modbase::SimplexReadPolicy policy(read);
    CATCH_CHECK(read.base_mod_simplex_motif_hits.size() == SEQUENCE.size());

    CATCH_REQUIRE(policy.next_strand());
    CATCH_CHECK(policy.is_template_direction());
    CATCH_CHECK(policy.target_start() == 0);
    CATCH_CHECK(policy.sequence() == SEQUENCE);
    CATCH_CHECK(policy.moves() == read.moves);
    CATCH_CHECK(policy.signal().equal(read.raw_data));
    CATCH_CHECK(policy.to_read_index(7) == 7);
    CATCH_CHECK_FALSE(policy.next_strand());
}

CATCH_TEST_CASE(TEST_GROUP ": simplex policy marks motif hits", TEST_GROUP) {
    ReadCommon read;
    read.seq = SEQUENCE;

    modbase::SimplexReadPolicy policy(read);
    policy.mark_motif_hit(3);
    policy.mark_motif_hit(10);

    for (size_t i = 0; i < SEQUENCE.size(); ++i) {
        CATCH_CAPTURE(i);
        CATCH_CHECK(read.base_mod_simplex_motif_hits[i] == (i == 3 || i == 10));
    }
    CATCH_CHECK_THROWS(policy.mark_motif_hit(SEQUENCE.size()));
}

CATCH_TEST_CASE(TEST_GROUP ": duplex policy yields template then complement", TEST_GROUP) {
    auto read = make_duplex_read();
    modbase::DuplexReadPolicy policy(read, STRIDE);

    // The realigned sequence is a window of the duplex sequence (in the strand's own direction)
    // starting at `target_start`, which the aligner may place after the first base.
    const auto check_strand = [&](const std::string& strand_sequence, const bool is_template) {
        CATCH_REQUIRE(policy.next_strand());
        CATCH_CAPTURE(is_template, policy.target_start());
        CATCH_CHECK(policy.is_template_direction() == is_template);

        const auto target_start = static_cast<size_t>(policy.target_start());
        CATCH_CHECK(policy.sequence() ==
                    strand_sequence.substr(target_start, policy.sequence().size()));
        CATCH_CHECK(policy.signal().size(0) ==
                    static_cast<int64_t>(policy.moves().size() * STRIDE));

        const size_t hit = 5;
        const size_t expected_index =
                is_template ? hit + target_start : SEQUENCE.size() - (hit + target_start + 1);
        CATCH_CHECK(policy.to_read_index(hit) == expected_index);
    };

    check_strand(SEQUENCE, true);
    check_strand(utils::reverse_complement(SEQUENCE), false);

    CATCH_CHECK_FALSE(policy.next_strand());
}

CATCH_TEST_CASE(TEST_GROUP ": duplex policy skips strands which cannot be realigned", TEST_GROUP) {
    auto read = make_duplex_read();
    // Neither strand aligns to the duplex sequence.
    read.stereo_feature_inputs.template_seq = "ACGT";
    read.stereo_feature_inputs.template_moves = one_move_per_base(4);
    read.read_common.seq = "TGAC";
    read.stereo_feature_inputs.complement_seq = "ACGT";
    read.stereo_feature_inputs.complement_moves = one_move_per_base(4);

    modbase::DuplexReadPolicy policy(read, STRIDE);
    CATCH_CHECK_FALSE(policy.next_strand());
}

CATCH_TEST_CASE(TEST_GROUP ": duplex policy marking motif hits does nothing", TEST_GROUP) {
    auto read = make_duplex_read();
    modbase::DuplexReadPolicy policy(read, STRIDE);
    policy.mark_motif_hit(3);
    CATCH_CHECK(read.read_common.base_mod_simplex_motif_hits.empty());
}
