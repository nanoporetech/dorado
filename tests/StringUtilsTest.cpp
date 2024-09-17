#include "utils/string_utils.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::string_utils]"

#include <string>
#include <string_view>

namespace dorado::utils::string_view {

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

TEST_CASE(CUT_TAG " join", CUT_TAG) {
    // clang-format off
    auto [inputs, separator, expected_result] = GENERATE(
            table<std::vector<std::string>, std::string, std::string >({
            std::make_tuple(std::vector<std::string>{""}, ",", ""),
            std::make_tuple(std::vector<std::string>{"test"}, " ", "test"),
            std::make_tuple(std::vector<std::string>{"a", "b", "c"}, "", "abc"),
            std::make_tuple(std::vector<std::string>{"a", "b", "c"}, " ", "a b c"),
            std::make_tuple(std::vector<std::string>{"word1;word2"}, ",", "word1;word2"),
            std::make_tuple(std::vector<std::string>{"word1", "word2", "word3"}, "; ", "word1; word2; word3"),
    }));
    // clang-format on

    CAPTURE(inputs);
    CAPTURE(separator);
    auto joined = dorado::utils::join(inputs, separator);
    CHECK(joined == expected_result);
}

TEST_CASE(CUT_TAG " starts_with") {
    // clang-format off
    auto [input, prefix, expected_results] = GENERATE(
        table<std::string, std::string, bool>({
            std::make_tuple("",       "",       true),
            std::make_tuple("aaa",    "bbb",    false),
            std::make_tuple("word",   "word",   true),
            std::make_tuple("word",   "rd",     false),
            std::make_tuple("word",   "",       true),
            std::make_tuple("word",   "vor",    false),
            std::make_tuple("word",   " wor",   false),
        })
    );
    // clang-format on

    CAPTURE(input);
    CAPTURE(prefix);
    auto starts_with = dorado::utils::starts_with(input, prefix);
    CHECK(starts_with == expected_results);
}

TEST_CASE(CUT_TAG " ends_with") {
    // clang-format off
    auto [input, suffix, expected_results] = GENERATE(
        table<std::string, std::string, bool>({
            std::make_tuple("",       "",       true),
            std::make_tuple("aaa",    "bbb",    false),
            std::make_tuple("word",   "word",   true),
            std::make_tuple("word",   "rd",     true),
            std::make_tuple("word",   "",       true),
            std::make_tuple("word",   "orc",    false),
            std::make_tuple("word",   "ord ",   false),
        })
    );
    // clang-format on

    CAPTURE(input);
    CAPTURE(suffix);
    auto starts_with = dorado::utils::ends_with(input, suffix);
    CHECK(starts_with == expected_results);
}

TEST_CASE(CUT_TAG " rtrim_view", CUT_TAG) {
    // clang-format off
    auto [input, expected] = GENERATE(
            table<std::string, std::string_view>({
            {"", ""},
            {"a", "a"},
            {"abc", "abc"},
            {"def  ", "def"},
            {"  \t   \t  ", ""},
            {"  \t def  \t  \t", "  \t def"},
            {"abc  \t\n  z", "abc  \t\n  z"},
            {"abc  \t\n  z  ", "abc  \t\n  z"},
    }));
    // clang-format on

    CAPTURE(input);

    auto actual = rtrim_view(input);

    REQUIRE(expected == actual);
}

}  // namespace dorado::utils::string_view