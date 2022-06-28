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

    RemoraEncoder encoder(BLOCK_STRIDE, SLICE_BLOCKS, 1, 1);
    encoder.encode_remora_data(moves, sequence);
    const auto& sample_offsets = encoder.get_sample_offsets();
    const auto& encoded_data = encoder.get_encoded_data();

    CHECK(sequence.size() == sample_offsets.size());
    size_t EXPECTED_ENCODING_LEN = (2 * PADDING + moves.size()) * BLOCK_STRIDE * KMER_LEN * 4;
    size_t encoded_data_size = 1;
    for (auto size : encoded_data.sizes()) {
        encoded_data_size *= size;
    }
    CHECK(EXPECTED_ENCODING_LEN == encoded_data_size);

    std::vector<int> expected_sample_offsets{0, 2, 6, 12, 14, 18, 22, 28, 32, 34};
    CHECK(expected_sample_offsets == sample_offsets);

    std::vector<float> expected_encoding_data{
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // NNT
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // NNT
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // NTA
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // NTA
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,  // TTC
            0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,  // TTC
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  // AGT
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // GTA
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // GTA
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // GTA
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // GTA
            0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,  // TAC
            0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,  // TAC
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // CNN
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // CNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
    };
    torch::Tensor expected_encoding = torch::from_blob(expected_encoding_data.data(), {52, 12});
    CHECK(torch::all(expected_encoding == encoded_data).item().toBool());

    auto slice0 = encoder.get_context(0);  // The T in the NTA 3mer.
    CHECK(slice0.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice0.first_sample == 0);
    CHECK(slice0.num_samples == 7);
    CHECK(slice0.lead_samples_needed == 5);
    CHECK(slice0.tail_samples_needed == 0);
    std::vector<float> expected_slice0_data = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // NNT
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  // NNT
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // NTA
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // NTA
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,  // TAT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
    };
    torch::Tensor expected_slice0 = torch::from_blob(expected_slice0_data.data(), {12, 12});
    CHECK(torch::all(expected_slice0 == slice0.data).item().toBool());

    auto slice1 = encoder.get_context(4);  // The C in the TCA 3mer.
    CHECK(slice1.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice1.first_sample == 10);
    CHECK(slice1.num_samples == 12);
    CHECK(slice1.lead_samples_needed == 0);
    CHECK(slice1.tail_samples_needed == 0);
    std::vector<float> expected_slice1_data = {
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,  // ATT
            0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,  // TTC
            0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,  // TTC
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,  // TCA
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  // CAG
    };
    torch::Tensor expected_slice1 = torch::from_blob(expected_slice1_data.data(), {12, 12});
    CHECK(torch::all(expected_slice1 == slice1.data).item().toBool());

    auto slice2 = encoder.get_context(9);  // The C in the ACN 3mer.
    CHECK(slice2.size == SLICE_BLOCKS * BLOCK_STRIDE * KMER_LEN * 4);
    CHECK(slice2.first_sample == 31);
    CHECK(slice2.num_samples == 9);
    CHECK(slice2.lead_samples_needed == 0);
    CHECK(slice2.tail_samples_needed == 3);
    std::vector<float> expected_slice2_data = {
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,  // GTA
            0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,  // TAC
            0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,  // TAC
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  // ACN
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // CNN
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // CNN
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // NNN
    };
    torch::Tensor expected_slice2 = torch::from_blob(expected_slice2_data.data(), {12, 12});
    CHECK(torch::all(expected_slice2 == slice2.data).item().toBool());
}