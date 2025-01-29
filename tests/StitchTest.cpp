#include "read_pipeline/stitch.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/math_utils.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[utils]"

// clang-format off
constexpr size_t RAW_SIGNAL_SIZE = 50;
const std::vector<std::string> SEQS(7, "ACGT");
const std::vector<std::string> QSTR(7, "!&.-");
const std::vector<std::vector<uint8_t>> MOVES{
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0}, 
        {1, 0, 0, 1, 0, 0, 0, 1, 0, 1},
        {1, 0, 0, 1, 0, 1, 1, 0, 0, 0},
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 0, 1, 0, 1, 0},
        {1, 0, 0, 0, 0, 0, 1, 0, 1, 1},
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0}};
/*
A        C        G     T
1, 0, 0, 1, 0, 0, 1, 0, 1, 0
                     A        C           G     T
                     1, 0, 0, 1, 0, 0, 0, 1, 0, 1
                                          A        C     G  T
                                          1, 0, 0, 1, 0, 1, 1, 0, 0, 0
                                                               A        C        G     T
                                                               1, 0, 0, 1, 0, 0, 1, 0, 1, 0
                                                                                       A     C        G     T
                                                                                    0, 1, 0, 1, 0, 0, 1, 0, 1, 0
                                                                                                         A                 C     G  T
                                                                                                         1, 0, 0, 0, 0, 0, 1, 0, 1, 1
                                                                                                                        A        C        G     T
                                                                                                                        1, 0, 0, 1, 0, 0, 1, 0, 1, 0
=
A        C        G     T     C           G        C     G  T           C        G     T     C        G     T              C     C        G     T
1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0
*/
// clang-format on
CATCH_TEST_CASE("Test stitch_chunks", TEST_GROUP) {
    constexpr size_t CHUNK_SIZE = 10;
    constexpr size_t OVERLAP = 3;

    std::vector<std::unique_ptr<dorado::utils::Chunk>> called_chunks;

    size_t offset = 0;
    size_t signal_chunk_step = CHUNK_SIZE - OVERLAP;
    {
        auto chunk = std::make_unique<dorado::utils::Chunk>(offset, CHUNK_SIZE);
        const size_t chunk_idx = called_chunks.size();
        chunk->qstring = QSTR[chunk_idx];
        chunk->seq = SEQS[chunk_idx];
        chunk->moves = MOVES[chunk_idx];
        called_chunks.push_back(std::move(chunk));
    }
    while (offset + CHUNK_SIZE < RAW_SIGNAL_SIZE) {
        offset = std::min(offset + signal_chunk_step, RAW_SIGNAL_SIZE - CHUNK_SIZE);
        auto chunk = std::make_unique<dorado::utils::Chunk>(offset, CHUNK_SIZE);
        const size_t chunk_idx = called_chunks.size();
        chunk->qstring = QSTR[chunk_idx];
        chunk->seq = SEQS[chunk_idx];
        chunk->moves = MOVES[chunk_idx];
        called_chunks.push_back(std::move(chunk));
    }

    dorado::ReadCommon read_common;
    read_common.model_stride = static_cast<int>(dorado::utils::div_round_closest(
            called_chunks[0]->raw_chunk_size, called_chunks[0]->moves.size()));
    CATCH_REQUIRE_NOTHROW(dorado::utils::stitch_chunks(read_common, called_chunks));

    const std::string expected_sequence = "ACGTCGCGTCGTCGTCCGT";
    const std::string expected_qstring = "!&.-&.&.-&.-&.-&&.-";
    const std::vector<uint8_t> expected_moves = {1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
                                                 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                                                 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1};

    CATCH_REQUIRE(read_common.seq == expected_sequence);
    CATCH_REQUIRE(read_common.qstring == expected_qstring);
    CATCH_REQUIRE(read_common.moves == expected_moves);
}
