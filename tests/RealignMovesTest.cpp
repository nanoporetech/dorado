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

TEST_CASE("Test realign_moves - output moves correct", TEST_GROUP) {
    SECTION("Test move table realignemnt of long identical sequences") {
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

        CHECK(old_moves_offset == 0);
        CHECK(query_start == 0);
        CHECK(new_moves.size() == input_moves.size());
        CHECK(new_moves == input_moves);
    }

    SECTION("Test move table realignemnt where target is suffix of query") {
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

        CHECK(old_moves_offset == 8);
        CHECK(target_start == 0);
        CHECK(new_moves.size() == input_moves.size() - 4 * 2);
        CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin() + 4 * 2));
    }

    SECTION("Test 2  -  move table realignemnt where target is suffix of query") {
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

        CHECK(old_moves_offset == 4 * 3);
        CHECK(target_start == 0);
        CHECK(new_moves.size() == input_moves.size() - 4 * 3);
        CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin() + 4 * 3));
    }

    SECTION("Test move table realignemnt where query is suffix of target") {
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

        CHECK(old_moves_offset == 0);
        CHECK(target_start == 5);
        CHECK(new_moves.size() == input_moves.size());
        CHECK(std::equal(new_moves.begin(), new_moves.end(), input_moves.begin()));
    }

    SECTION("Test move table realignemnt where target is an infix of query") {
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

        CHECK(old_moves_offset == 5 * 4);
        CHECK(target_start == 0);
        CHECK(new_moves.size() == input_moves.size() - (5 + 5) * 4);
    }

    SECTION("Test move table realignemnt where query is an infix of target") {
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

        CHECK(old_moves_offset == 0);
        CHECK(target_start == 5);
        CHECK(new_moves.size() == input_moves.size());
    }
}