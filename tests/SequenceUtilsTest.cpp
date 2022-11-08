#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[utils]"

using namespace dorado::utils;

TEST_CASE(TEST_GROUP ": Test base_to_int") {
    CHECK(base_to_int('A') == 0);
    CHECK(base_to_int('C') == 1);
    CHECK(base_to_int('G') == 2);
    CHECK(base_to_int('T') == 3);
}

TEST_CASE(TEST_GROUP ": Test sequence_to_ints") {
    SECTION("Test empty string") {
        auto actual_results = sequence_to_ints("");
        std::vector<int> expected_results = {};
        CHECK(expected_results == actual_results);
    }

    SECTION("Test single char") {
        auto actual_results = sequence_to_ints("G");
        std::vector<int> expected_results = {2};
        CHECK(expected_results == actual_results);
    }
}