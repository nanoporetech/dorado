#include "utils/string_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#define CUT_TAG "[dorado::utils::string_utils]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

#include <string>
#include <string_view>

namespace dorado::utils::string_view {

DEFINE_TEST("split") {
    // clang-format off
    auto [input, delimiter, expected_results] = GENERATE(
        table<std::string, char, std::vector<std::string>>({
            std::make_tuple("",            ',', std::vector<std::string>{""}),
            std::make_tuple("test",        ',', std::vector<std::string>{"test"}),
            std::make_tuple("word1;word2", ',', std::vector<std::string>{"word1;word2"}),
            std::make_tuple("word1;word2", ';', std::vector<std::string>{"word1", "word2"}),
            std::make_tuple("word1;word2;", ';', std::vector<std::string>{"word1", "word2", ""}),
            std::make_tuple(",", ',', std::vector<std::string>{"", ""}),
        })
    );
    // clang-format on

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(delimiter);
    auto tokens = dorado::utils::split(input, delimiter);
    CATCH_CHECK(tokens == expected_results);
}

DEFINE_TEST("split_view") {
    // clang-format off
    auto [input, delimiter, expected_results] = GENERATE(
        table<std::string_view, char, std::vector<std::string_view>>({
            std::make_tuple("",            ',', std::vector<std::string_view>{}),
            std::make_tuple("test",        ',', std::vector<std::string_view>{"test"}),
            std::make_tuple("word1;word2", ',', std::vector<std::string_view>{"word1;word2"}),
            std::make_tuple("word1;word2", ';', std::vector<std::string_view>{"word1", "word2"}),
            std::make_tuple("word1;word2;", ';', std::vector<std::string_view>{"word1", "word2", ""}),
            std::make_tuple(",", ',', std::vector<std::string_view>{"", ""}),
        })
    );
    // clang-format on

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(delimiter);
    const std::vector<std::string_view> tokens = dorado::utils::split_view(input, delimiter);
    CATCH_CHECK(tokens == expected_results);
}

DEFINE_TEST("join") {
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

    CATCH_CAPTURE(inputs);
    CATCH_CAPTURE(separator);
    auto joined = dorado::utils::join(inputs, separator);
    CATCH_CHECK(joined == expected_result);
}

DEFINE_TEST("starts_with") {
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

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(prefix);
    auto starts_with = dorado::utils::starts_with(input, prefix);
    CATCH_CHECK(starts_with == expected_results);
}

DEFINE_TEST("ends_with") {
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

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(suffix);
    auto starts_with = dorado::utils::ends_with(input, suffix);
    CATCH_CHECK(starts_with == expected_results);
}

DEFINE_TEST("contains") {
    auto [input, substr, expected_result] = GENERATE(table<std::string, std::string, bool>({
            std::make_tuple("", "", true),
            std::make_tuple("aaa", "bbb", false),
            std::make_tuple("word", "word", true),
            std::make_tuple("word", "", true),
            std::make_tuple("word", "or", true),
            std::make_tuple("word", "or ", false),
            std::make_tuple("or", "word", false),
    }));

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(substr);
    auto starts_with = dorado::utils::contains(input, substr);
    CATCH_CHECK(starts_with == expected_result);
}

DEFINE_TEST("rtrim_view") {
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
    CATCH_CAPTURE(input);

    auto actual = rtrim_view(input);

    CATCH_REQUIRE(expected == actual);
}

}  // namespace dorado::utils::string_view