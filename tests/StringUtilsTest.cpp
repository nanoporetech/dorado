#include "utils/string_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#define CUT_TAG "[dorado::utils::string_utils]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::string_view {

DEFINE_TEST("split") {
    auto [input, delimiter,
          expected_results] = GENERATE(table<std::string, char, std::vector<std::string>>({
            std::make_tuple("", ',', std::vector<std::string>{""}),
            std::make_tuple("test", ',', std::vector<std::string>{"test"}),
            std::make_tuple("word1;word2", ',', std::vector<std::string>{"word1;word2"}),
            std::make_tuple("word1;word2", ';', std::vector<std::string>{"word1", "word2"}),
            std::make_tuple("word1;word2;", ';', std::vector<std::string>{"word1", "word2", ""}),
            std::make_tuple(",", ',', std::vector<std::string>{"", ""}),
    }));

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(delimiter);
    auto tokens = dorado::utils::split(input, delimiter);
    CATCH_CHECK(tokens == expected_results);
}

DEFINE_TEST("split_view") {
    auto [input, delimiter, expected_results] =
            GENERATE(table<std::string_view, char, std::vector<std::string_view>>({
                    std::make_tuple("", ',', std::vector<std::string_view>{}),
                    std::make_tuple("test", ',', std::vector<std::string_view>{"test"}),
                    std::make_tuple("word1;word2", ',',
                                    std::vector<std::string_view>{"word1;word2"}),
                    std::make_tuple("word1;word2", ';',
                                    std::vector<std::string_view>{"word1", "word2"}),
                    std::make_tuple("word1;word2;", ';',
                                    std::vector<std::string_view>{"word1", "word2", ""}),
                    std::make_tuple(",", ',', std::vector<std::string_view>{"", ""}),
            }));

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(delimiter);
    const std::vector<std::string_view> tokens = dorado::utils::split_view(input, delimiter);
    CATCH_CHECK(tokens == expected_results);
}

DEFINE_TEST("join") {
    auto [inputs, separator, expected_result] =
            GENERATE(table<std::vector<std::string>, std::string, std::string>({
                    std::make_tuple(std::vector<std::string>{""}, ",", ""),
                    std::make_tuple(std::vector<std::string>{"test"}, " ", "test"),
                    std::make_tuple(std::vector<std::string>{"a", "b", "c"}, "", "abc"),
                    std::make_tuple(std::vector<std::string>{"a", "b", "c"}, " ", "a b c"),
                    std::make_tuple(std::vector<std::string>{"word1;word2"}, ",", "word1;word2"),
                    std::make_tuple(std::vector<std::string>{"word1", "word2", "word3"}, "; ",
                                    "word1; word2; word3"),
            }));

    CATCH_CAPTURE(inputs);
    CATCH_CAPTURE(separator);
    auto joined = dorado::utils::join(inputs, separator);
    CATCH_CHECK(joined == expected_result);
}

DEFINE_TEST("starts_with") {
    auto [input, prefix, expected_results] = GENERATE(table<std::string, std::string, bool>({
            std::make_tuple("", "", true),
            std::make_tuple("aaa", "bbb", false),
            std::make_tuple("word", "word", true),
            std::make_tuple("word", "rd", false),
            std::make_tuple("word", "", true),
            std::make_tuple("word", "vor", false),
            std::make_tuple("word", " wor", false),
    }));

    CATCH_CAPTURE(input);
    CATCH_CAPTURE(prefix);
    auto starts_with = dorado::utils::starts_with(input, prefix);
    CATCH_CHECK(starts_with == expected_results);
}

DEFINE_TEST("ends_with") {
    auto [input, suffix, expected_results] = GENERATE(table<std::string, std::string, bool>({
            std::make_tuple("", "", true),
            std::make_tuple("aaa", "bbb", false),
            std::make_tuple("word", "word", true),
            std::make_tuple("word", "rd", true),
            std::make_tuple("word", "", true),
            std::make_tuple("word", "orc", false),
            std::make_tuple("word", "ord ", false),
    }));

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
    auto [input, expected] = GENERATE(table<std::string, std::string_view>({
            {"", ""},
            {"a", "a"},
            {"abc", "abc"},
            {"def  ", "def"},
            {"  \t   \t  ", ""},
            {"  \t def  \t  \t", "  \t def"},
            {"abc  \t\n  z", "abc  \t\n  z"},
            {"abc  \t\n  z  ", "abc  \t\n  z"},
    }));

    CATCH_CAPTURE(input);

    auto actual = rtrim_view(input);

    CATCH_REQUIRE(expected == actual);
}

DEFINE_TEST("from_chars") {
    CATCH_CHECK(from_chars<int>("123") == 123);
    CATCH_CHECK(from_chars<std::int64_t>("-10") == -10);

    CATCH_CHECK(from_chars<int>("not a number") == std::nullopt);
    CATCH_CHECK(from_chars<std::uint8_t>("256") == std::nullopt);
}

template <typename T>
void test_to_chars_for_type() {
    CATCH_CAPTURE(typeid(T).name());

    auto check_value = [](auto val) {
        const auto input = static_cast<T>(val);
        const std::string expected = std::is_integral_v<T> ? std::to_string(input)
                                                           : (std::ostringstream{} << input).str();
        CATCH_CAPTURE(val, input, expected);

        // With a temporary buffer.
        {
            const auto result = to_chars(input);
            CATCH_CHECK(result.view() == expected);
        }

        // Into a provided buffer, deduced size.
        {
            char buffer[128]{};
            const std::string_view view = to_chars(input, buffer);
            CATCH_CHECK(view.size() <= std::size(buffer));
            CATCH_CHECK(view == expected);
            // Raw data should match.
            CATCH_CHECK(std::string_view{buffer, view.size()} == expected);
        }

        // Into a provided buffer, raw pointer.
        {
            constexpr std::size_t Size = 128;
            std::vector<char> buffer(Size);
            const std::string_view view = to_chars<Size>(input, buffer.data());
            CATCH_CHECK(view.size() <= Size);
            CATCH_CHECK(view == expected);
            // Raw data should match.
            CATCH_CHECK(std::string_view{buffer.data(), view.size()} == expected);
        }
    };

    using Limits = std::numeric_limits<T>;
    check_value(0);
    check_value(1);
    check_value(-1);
    check_value(42);
    if constexpr (std::is_integral_v<T>) {
        check_value(Limits::max());
        check_value(Limits::max() - 1);
        check_value(Limits::min());
        check_value(Limits::min() + 1);
    } else {
        check_value(5.5);
        check_value(0.01);
        check_value(Limits::quiet_NaN());
        // Only check that these don't throw since different implementations will give different precision.
        CATCH_CHECK_NOTHROW(to_chars(Limits::max()));
        CATCH_CHECK_NOTHROW(to_chars(Limits::max() - 1));
        CATCH_CHECK_NOTHROW(to_chars(Limits::min()));
        CATCH_CHECK_NOTHROW(to_chars(Limits::min() + 1));
        CATCH_CHECK_NOTHROW(to_chars(Limits::lowest()));
    }
}
DEFINE_TEST("to_chars") {
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::int8_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::int16_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::int32_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::int64_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::uint8_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::uint16_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::uint32_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<std::uint64_t>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<float>());
    CATCH_CHECK_NOTHROW(test_to_chars_for_type<double>());
}

}  // namespace dorado::utils::string_view
