#include "utils/string_utils.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[string_utils]"

TEST_CASE(CUT_TAG " split", CUT_TAG) {
    // clang-format off
    auto [input, delimiter, expected_results] = GENERATE(
        table<std::string, char, std::vector<std::string>>({
            std::make_tuple("",            ',', std::vector<std::string>{""}),
            std::make_tuple("test",        ',', std::vector<std::string>{"test"}),
            std::make_tuple("word1;word2", ',', std::vector<std::string>{"word1;word2"}),
            std::make_tuple("word1;word2", ';', std::vector<std::string>{"word1", "word2"})
        })
    );
    // clang-format on

    CAPTURE(input);
    CAPTURE(delimiter);
    auto tokens = dorado::utils::split(input, delimiter);
    CHECK(tokens == expected_results);
}
