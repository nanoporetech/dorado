#include "modbase/remora_encoder.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[remora_encoder]"

TEST_CASE("Encode sequence for modified basecalling", TEST_GROUP) {
    const size_t BLOCK_STRIDE = 2;
    const size_t KMER_LEN = 3;
    const size_t SLICE_BLOCKS = 6;
    const size_t PADDING = SLICE_BLOCKS / 2;
    std::string sequence{"TATTCAGTAC"};
    //                         T  A     T        T  C     A     G        T     A  C
    std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};

    RemoraEncoder encoder(BLOCK_STRIDE, SLICE_BLOCKS * BLOCK_STRIDE, 1, 1);
    encoder.encode_remora_data(moves, sequence);
    const auto& sample_offsets = encoder.get_sample_offsets();

    CHECK(sequence.size() == sample_offsets.size());

    std::vector<int> expected_sample_offsets{0, 2, 6, 12, 14, 18, 22, 28, 32, 34};
    CHECK(expected_sample_offsets == sample_offsets);

    auto slice0 = encoder.get_context(0);  // The T in the NTA 3mer.
    CHECK(slice0.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice0.first_sample == 0);
    CHECK(slice0.num_samples == 7);
    CHECK(slice0.lead_samples_needed == 5);
    CHECK(slice0.tail_samples_needed == 0);

    // clang-format off
    std::vector<float> expected_slice0 = {
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,
            1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
        // NTA NTA NTA NTA NTA NTA NTA TAT TAT TAT TAT ATT
    };    
    // clang-format on    
    CHECK(expected_slice0 == slice0.data);

    auto slice1 = encoder.get_context(4);  // The C in the TCA 3mer.
    CHECK(slice1.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice1.first_sample == 10);
    CHECK(slice1.num_samples == 12);
    CHECK(slice1.lead_samples_needed == 0);
    CHECK(slice1.tail_samples_needed == 0);

// clang-format off
    std::vector<float> expected_slice1 = {
            1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
            0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,
            0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
            1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        // ATT ATT TTC TTC TCA TCA TCA TCA CAG CAG CAG CAG
    };
    // clang-format on
    CHECK(expected_slice1 == slice1.data);

    auto slice2 = encoder.get_context(9);  // The C in the ACN 3mer.
    CHECK(slice2.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice2.first_sample == 31);
    CHECK(slice2.num_samples == 9);
    CHECK(slice2.lead_samples_needed == 0);
    CHECK(slice2.tail_samples_needed == 3);

    // clang-format off
    std::vector<float> expected_slice2 = {
            0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        // GTA TAC TAC ACN ACN ACN ACN ACN ACN ACN ACN ACN
    };
    // clang-format on    
    CHECK(expected_slice2 == slice2.data);
}