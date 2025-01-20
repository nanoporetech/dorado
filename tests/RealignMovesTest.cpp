#include "utils/sequence_utils.h"

#include <ATen/TensorIndexing.h>
#include <catch2/catch_test_macros.hpp>

using Slice = at::indexing::Slice;
using namespace dorado;

#define TEST_GROUP "[utils][realign_moves]"

CATCH_TEST_CASE("Realign Moves No Error", TEST_GROUP) {
    std::string query_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";   // Example query sequence
    std::string target_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";  // Example target sequence
    std::vector<uint8_t> moves = {
            1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
            0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0,
            1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1};  // Example moves vector

    // Test that calling realign_moves does not throw any exceptions
    CATCH_CHECK_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));
}

CATCH_TEST_CASE("No alignment doesn't produce an error", TEST_GROUP) {
    std::string query_sequence = "ACGT";                    // Example query sequence
    std::string target_sequence = "TGAC";                   // Example target sequence
    std::vector<uint8_t> moves = {1, 0, 1, 0, 1, 0, 0, 1};  // Original moves vector

    // Check that the function does not throw an exception
    CATCH_REQUIRE_NOTHROW(utils::realign_moves(query_sequence, target_sequence, moves));

    // Call the function and store the result
    int move_offset = 0, target_start = 0;
    std::vector<uint8_t> new_moves;

    CATCH_CHECK_NOTHROW(std::tie(move_offset, target_start, new_moves) =
                                utils::realign_moves(query_sequence, target_sequence, moves));

    CATCH_CHECK(move_offset == -1);
    CATCH_CHECK(target_start == -1);
    CATCH_CHECK(new_moves.empty());
}

CATCH_TEST_CASE("Test realign_moves - output moves correct", TEST_GROUP) {
    CATCH_SECTION("Test move table realignment of long identical sequences") {
        auto query =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, query_start, new_moves] =
                utils::realign_moves(query, target, input_moves);

        CATCH_CHECK(old_moves_offset == 0);
        CATCH_CHECK(query_start == 0);
        CATCH_CHECK(new_moves.size() == input_moves.size());
        CATCH_CHECK(new_moves == input_moves);
    }

    CATCH_SECTION("Test move table realignment where target is suffix of query") {
        auto query = std::string(
                "TTTTACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, target_start, new_moves] =
                utils::realign_moves(query, target, input_moves);  // simplex, duplex, moves

        CATCH_CHECK(old_moves_offset == 8);
        CATCH_CHECK(target_start == 0);
        CATCH_CHECK(new_moves.size() == input_moves.size() - 4 * 2);
        CATCH_CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin() + 4 * 2));
    }

    CATCH_SECTION("Test 2  -  move table realignment where target is suffix of query") {
        auto query = std::string(
                "TTTTACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, target_start, new_moves] =
                utils::realign_moves(query, target, input_moves);  // simplex, duplex, moves

        CATCH_CHECK(old_moves_offset == 4 * 3);
        CATCH_CHECK(target_start == 0);
        CATCH_CHECK(new_moves.size() == input_moves.size() - 4 * 3);
        CATCH_CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin() + 4 * 3));
    }

    CATCH_SECTION("Test move table realignment where query is suffix of target") {
        auto query =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target = std::string(
                "TTTTTACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
            input_moves.push_back(0);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, target_start, new_moves] =
                utils::realign_moves(query, target, input_moves);  // simplex, duplex, moves

        CATCH_CHECK(old_moves_offset == 0);
        CATCH_CHECK(target_start == 5);
        CATCH_CHECK(new_moves.size() == input_moves.size());
        CATCH_CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin()));
    }

    CATCH_SECTION("Test move table realignment where target is an infix of query") {
        auto query = std::string(
                "GGGGGACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTTGGGGG");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
            input_moves.push_back(0);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, target_start, new_moves] =
                utils::realign_moves(query, target, input_moves);  // simplex, duplex, moves

        CATCH_CHECK(old_moves_offset == 5 * 4);
        CATCH_CHECK(target_start == 0);
        CATCH_CHECK(new_moves.size() == input_moves.size() - (5 + 5) * 4);
    }

    CATCH_SECTION("Test move table realignment where query is an infix of target") {
        auto query =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target = std::string(
                "GGGGGACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTTGGGGG");

        std::vector<uint8_t> input_moves;

        for (char nucleotide : query) {
            (void)nucleotide;
            input_moves.push_back(1);
            input_moves.push_back(0);
            input_moves.push_back(0);
        }

        auto [old_moves_offset, target_start, new_moves] =
                utils::realign_moves(query, target, input_moves);  // simplex, duplex, moves

        CATCH_CHECK(old_moves_offset == 0);
        CATCH_CHECK(target_start == 5);
        CATCH_CHECK(new_moves.size() == input_moves.size());
    }
}