#include "modbase/ModbaseEncoder.h"

#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[modbase_encoder]"

TEST_CASE("Encode sequence for modified basecalling", TEST_GROUP) {
    const size_t BLOCK_STRIDE = 2;
    const size_t SLICE_BLOCKS = 6;
    std::string sequence{"TATTCAGTAC"};
    auto seq_ints = dorado::utils::sequence_to_ints(sequence);
    //                         T  A     T        T  C     A     G        T     A  C
    std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
    auto seq_to_sig_map = dorado::utils::moves_to_map(moves, BLOCK_STRIDE,
                                                      moves.size() * BLOCK_STRIDE, std::nullopt);

    dorado::modbase::ModBaseEncoder encoder(BLOCK_STRIDE, SLICE_BLOCKS * BLOCK_STRIDE, 1, 1);
    encoder.init(seq_ints, seq_to_sig_map);

    auto slice0 = encoder.get_context(0);  // The T in the NTA 3mer.
    CHECK(slice0.first_sample == 0);
    CHECK(slice0.num_samples == 7);
    CHECK(slice0.lead_samples_needed == 5);
    CHECK(slice0.tail_samples_needed == 0);

    // clang-format off
    std::vector<int8_t> expected_slice0 = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
    };    
    // clang-format on    
    CHECK(expected_slice0 == slice0.data);

    auto slice1 = encoder.get_context(4);  // The C in the TCA 3mer.
    CHECK(slice1.first_sample == 10);
    CHECK(slice1.num_samples == 12);
    CHECK(slice1.lead_samples_needed == 0);
    CHECK(slice1.tail_samples_needed == 0);

    // clang-format off
    std::vector<int8_t> expected_slice1 = {
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, // TTC
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, // TTC
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
    };
    // clang-format on
    CHECK(expected_slice1 == slice1.data);

    auto slice2 = encoder.get_context(9);  // The C in the ACN 3mer.
    CHECK(slice2.first_sample == 31);
    CHECK(slice2.num_samples == 9);
    CHECK(slice2.lead_samples_needed == 0);
    CHECK(slice2.tail_samples_needed == 3);

    // clang-format off
    std::vector<int8_t> expected_slice2 = {
        0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, // GTA
        0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, // TAC
        0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, // TAC
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
    };
    // clang-format on    
    CHECK(expected_slice2 == slice2.data);
}
