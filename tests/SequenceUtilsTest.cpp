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

TEST_CASE(TEST_GROUP "reverse_complement") {
    REQUIRE(dorado::utils::reverse_complement("") == "");
    REQUIRE(dorado::utils::reverse_complement("ACGT") == "ACGT");
    std::srand(42);
    const std::string bases("ACGT");
    for (int i = 0; i < 10; ++i) {
        const int len = std::rand() % 20000;
        std::string temp(len, ' ');
        std::string rev_comp(len, ' ');
        for (int j = 0; j < len; ++j) {
            const int base_index = std::rand() % 4;
            temp.at(j) = bases.at(base_index);
            rev_comp.at(len - 1 - j) = bases.at(3 - base_index);
        }
        REQUIRE(dorado::utils::reverse_complement(temp) == rev_comp);
    }
}
