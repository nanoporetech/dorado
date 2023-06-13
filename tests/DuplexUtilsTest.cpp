#include "utils/duplex_utils.h"

#include <catch2/catch.hpp>

#include <cstdlib>

#define TEST_GROUP "[utils]"

using namespace dorado::utils;

TEST_CASE(TEST_GROUP ": Test grouping reads into completed duplex reads and simplex only reads") {
    std::unordered_set<std::string> read_set = {"c", "d", "a", "b", "a;b", "f", "e", "d;e"};
    std::unordered_set<std::string> completed_duplex;
    std::unordered_set<std::string> simplex;

    split_completed_duplex_reads(completed_duplex, simplex, read_set);

    CHECK(completed_duplex.find("a") != completed_duplex.end());
    CHECK(completed_duplex.find("b") != completed_duplex.end());
    CHECK(completed_duplex.find("d") != completed_duplex.end());
    CHECK(completed_duplex.find("e") != completed_duplex.end());
    CHECK(completed_duplex.size() == 4);

    CHECK(simplex.find("c") != simplex.end());
    CHECK(simplex.find("f") != simplex.end());
    CHECK(simplex.size() == 2);
}
