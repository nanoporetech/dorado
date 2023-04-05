#include "utils/duplex_utils.h"

#include <cstdlib>

#include <catch2/catch.hpp>

#define TEST_GROUP "DuplexUtils: "

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