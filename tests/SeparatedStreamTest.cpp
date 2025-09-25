#include "utils/SeparatedStream.h"

#include "catch2/catch_message.hpp"
#include "utils/string_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <optional>

#define TEST_GROUP "SeparatedStream"
#define DEFINE_TEST(name) CATCH_TEST_CASE(TEST_GROUP " : " name, TEST_GROUP)

template <char Separator>
void test_split_on_separator(bool stream_ops, bool with_trailing) {
    using namespace std::string_view_literals;

    std::vector inputs{"this"sv, "is"sv, "a"sv, "test"sv};
    if (with_trailing) {
        // A trailing element should be ignored.
        inputs.emplace_back();
    }

    const char separator_str[] = {Separator, '\0'};
    const auto input = dorado::utils::join(inputs, separator_str);

    dorado::utils::SeparatedStream<Separator> stream(input);
    CATCH_CHECK_FALSE(stream.eof());

    if (stream_ops) {
        std::string_view temp;
        CATCH_CHECK(stream.peek() == "this");
        CATCH_CHECK_FALSE((stream >> temp).eof());
        CATCH_CHECK(temp == "this");

        CATCH_CHECK(stream.peek() == "is");
        CATCH_CHECK_FALSE((stream >> temp).eof());
        CATCH_CHECK(temp == "is");

        CATCH_CHECK(stream.peek() == "a");
        CATCH_CHECK_FALSE((stream >> temp).eof());
        CATCH_CHECK(temp == "a");

        CATCH_CHECK(stream.peek() == "test");
        CATCH_CHECK_FALSE((stream >> temp).eof());
        CATCH_CHECK(temp == "test");

        CATCH_CHECK(stream.peek() == std::nullopt);
        CATCH_CHECK((stream >> temp).eof());
    } else {
        CATCH_CHECK(stream.peek() == "this");
        CATCH_CHECK(stream.getline() == "this");
        CATCH_CHECK_FALSE(stream.eof());

        CATCH_CHECK(stream.peek() == "is");
        CATCH_CHECK(stream.getline() == "is");
        CATCH_CHECK_FALSE(stream.eof());

        CATCH_CHECK(stream.peek() == "a");
        CATCH_CHECK(stream.getline() == "a");
        CATCH_CHECK_FALSE(stream.eof());

        CATCH_CHECK(stream.peek() == "test");
        CATCH_CHECK(stream.getline() == "test");
        CATCH_CHECK_FALSE(stream.eof());

        CATCH_CHECK(stream.peek() == std::nullopt);
        CATCH_CHECK(stream.getline() == std::nullopt);
        CATCH_CHECK(stream.eof());
    }
}

DEFINE_TEST("split on separator") {
    const bool stream_ops = GENERATE(true, false);
    const bool with_trailing = GENERATE(true, false);
    CATCH_CAPTURE(stream_ops, with_trailing);

    test_split_on_separator<'\n'>(stream_ops, with_trailing);
    test_split_on_separator<'\t'>(stream_ops, with_trailing);
    test_split_on_separator<'_'>(stream_ops, with_trailing);
}

DEFINE_TEST("no separator") {
    const bool stream_ops = GENERATE(true, false);
    CATCH_CAPTURE(stream_ops);

    const std::string_view input = "no tabs";
    dorado::utils::TabSeparatedStream stream(input);
    CATCH_CHECK_FALSE(stream.eof());

    if (stream_ops) {
        std::string_view temp;
        CATCH_CHECK_FALSE((stream >> temp).eof());
        CATCH_CHECK(temp == input);

        CATCH_CHECK((stream >> temp).eof());
    } else {
        CATCH_CHECK(stream.getline() == input);
        CATCH_CHECK_FALSE(stream.eof());

        CATCH_CHECK(stream.getline() == std::nullopt);
        CATCH_CHECK(stream.eof());
    }
}

DEFINE_TEST("empty lines") {
    const bool stream_ops = GENERATE(true, false);
    const int num_lines = GENERATE(0, 1, 2, 3);
    CATCH_CAPTURE(stream_ops, num_lines);

    const std::string input(num_lines, '\n');
    dorado::utils::NewlineSeparatedStream stream(input);
    CATCH_CHECK_FALSE(stream.eof());

    if (stream_ops) {
        std::string_view temp;
        for (int i = 0; i < num_lines; i++) {
            CATCH_CHECK(stream.peek() == "");
            CATCH_CHECK_FALSE((stream >> temp).eof());
            CATCH_CHECK(temp == "");
        }

        CATCH_CHECK(stream.peek() == std::nullopt);
        CATCH_CHECK((stream >> temp).eof());
    } else {
        for (int i = 0; i < num_lines; i++) {
            CATCH_CHECK(stream.peek() == "");
            CATCH_CHECK(stream.getline() == "");
            CATCH_CHECK_FALSE(stream.eof());
        }

        CATCH_CHECK(stream.peek() == std::nullopt);
        CATCH_CHECK(stream.getline() == std::nullopt);
        CATCH_CHECK(stream.eof());
    }
}

DEFINE_TEST("stream operators") {
    const char input[] = "test 123 -2  blah";

    std::string_view test;
    uint64_t num_123 = 0;
    int8_t num_negative_2 = 0;
    std::string_view empty = "not empty";
    std::string_view blah;

    dorado::utils::SpaceSeparatedStream stream(input);
    CATCH_CHECK_FALSE(stream.eof());

    stream >> test >> num_123 >> num_negative_2 >> empty >> blah;
    CATCH_CHECK_FALSE(stream.eof());

    CATCH_CHECK(test == "test");
    CATCH_CHECK(num_123 == 123);
    CATCH_CHECK(num_negative_2 == -2);
    CATCH_CHECK(empty == "");
    CATCH_CHECK(blah == "blah");
}
