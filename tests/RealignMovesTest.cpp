#include "utils/sequence_utils.h"

#include <ATen/TensorIndexing.h>
#include <catch2/catch.hpp>

using Slice = at::indexing::Slice;
using namespace dorado;

#define TEST_GROUP "[utils][realign_moves]"

TEST_CASE("Realign Moves No Error", TEST_GROUP) {
    std::string query_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";   // Example query sequence
    std::string target_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";  // Example target sequence
    std::vector<uint8_t> moves = {
            1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
            0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0,
            1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1};  // Example moves vector

    // Test that calling realign_moves does not throw any exceptions
    CHECK_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));
}

TEST_CASE("No alignment doesn't produce an error", TEST_GROUP) {
    std::string query_sequence = "ACGT";                    // Example query sequence
    std::string target_sequence = "TGAC";                   // Example target sequence
    std::vector<uint8_t> moves = {1, 0, 1, 0, 1, 0, 0, 1};  // Original moves vector

    // Check that the function does not throw an exception
    REQUIRE_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));

    // Call the function and store the result
    int move_offset, target_start;
    std::vector<uint8_t> new_moves;

    CHECK_NOTHROW(std::tie(move_offset, target_start, new_moves) =
                          utils::realign_moves(query_sequence, target_sequence, moves));

    CHECK(move_offset == -1);
    CHECK(target_start == -1);
    CHECK(new_moves.empty());
}