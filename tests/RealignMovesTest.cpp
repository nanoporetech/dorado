#include "utils/sequence_utils.h"

#include <ATen/ATen.h>
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