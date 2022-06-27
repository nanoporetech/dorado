#include "modbase/remora_scaler.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[remora_scaler]"

TEST_CASE(TEST_GROUP " scaler seq_to_ints", TEST_GROUP) {
    SECTION("Test good sequence") {
        std::string sequence{"TATTCAGTAC"};
        std::vector<int> expected_int_sequence{3, 0, 3, 3, 1, 0, 2, 3, 0, 1};
        auto int_sequence = RemoraScaler::seq_to_ints(sequence);
        REQUIRE(int_sequence.size() == expected_int_sequence.size());
        for (size_t i = 0; i < int_sequence.size(); ++i) {
            REQUIRE(int_sequence[i] == expected_int_sequence[i]);
        }
    }

    SECTION("Test empty sequence") {
        auto empty_sequence = RemoraScaler::seq_to_ints({});
        REQUIRE(empty_sequence.empty());
    }

    SECTION("Test invalid sequence") {
        std::string invalid_sequence{"TATTCTNACT"};
        REQUIRE_THROWS_AS(RemoraScaler::seq_to_ints(invalid_sequence), std::invalid_argument);
    }
}

TEST_CASE(TEST_GROUP " scaler moves_to_map", TEST_GROUP) {
    constexpr size_t STRIDE = 5;
    constexpr size_t SIGNAL_LENGTH = 105;
    std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
    auto move_map = RemoraScaler::moves_to_map(moves, STRIDE, SIGNAL_LENGTH);

    std::vector<uint64_t> expected_move_map{0, 5, 15, 30, 35, 45, 55, 70, 80, 85, 105};
    REQUIRE(move_map.size() == expected_move_map.size());
    for (size_t i = 0; i < move_map.size(); ++i) {
        REQUIRE(move_map[i] == expected_move_map[i]);
    }
}
