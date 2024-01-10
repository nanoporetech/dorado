#include "utils/sequence_utils.h"

#include <ATen/ATen.h>
#include <catch2/catch.hpp>

using Slice = at::indexing::Slice;
using namespace dorado;

#define TEST_GROUP "[utils][realign_moves]"

TEST_CASE("Realign Moves No Error", TEST_GROUP) {
    std::string query_sequence = "ACGT";                    // Example query sequence
    std::string target_sequence = "ACGT";                   // Example target sequence
    std::vector<uint8_t> moves = {1, 0, 1, 0, 1, 0, 0, 1};  // Example moves vector

    // Test that calling realign_moves does not throw any exceptions
    REQUIRE_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));
}

TEST_CASE("No alignment doesn't produce an error", TEST_GROUP) {
    std::string query_sequence = "ACGT";                    // Example query sequence
    std::string target_sequence = "TGAC";                   // Example target sequence
    std::vector<uint8_t> moves = {1, 0, 1, 0, 1, 0, 0, 1};  // Original moves vector

    // Check that the function does not throw an exception
    REQUIRE_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));

    // Call the function and store the result
    auto result = utils::realign_moves(query_sequence, target_sequence, moves);

    // Check that the returned tuple has the expected values
    REQUIRE(std::get<0>(result) == -1);    // Check the first value of the tuple
    REQUIRE(std::get<1>(result) == -1);    // Check the second value of the tuple
    REQUIRE(std::get<2>(result).empty());  // Check that the vector in the tuple is empty
}