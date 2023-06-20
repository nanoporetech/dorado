#include "utils/cli_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[cli_utils]"

using namespace dorado::utils;

TEST_CASE("CliUtils: Check thread allocation", TEST_GROUP) {
    int aligner_threads, writer_threads;

    SECTION("fraction > 0") {
        std::tie(aligner_threads, writer_threads) = aligner_writer_thread_allocation(10, 0.25f);
        REQUIRE(aligner_threads == 8);
        REQUIRE(writer_threads == 2);
    }

    SECTION("fraction == 0") {
        std::tie(aligner_threads, writer_threads) = aligner_writer_thread_allocation(10, 0.0f);
        REQUIRE(aligner_threads == 9);
        REQUIRE(writer_threads == 1);
    }

    SECTION("fraction 100%") {
        std::tie(aligner_threads, writer_threads) = aligner_writer_thread_allocation(10, 1.0f);
        REQUIRE(aligner_threads == 1);
        REQUIRE(writer_threads == 9);
    }
}

TEST_CASE("CliUtils: Check number string to unsigned long conversion", TEST_GROUP) {
    SECTION("convert #K") { CHECK(parse_string_to_size("5K") == 5000); }
    SECTION("convert #K") { CHECK(parse_string_to_size("5K") == 5000); }
    SECTION("convert #M") { CHECK(parse_string_to_size("5M") == 5000000); }
    SECTION("convert #G") { CHECK(parse_string_to_size("5G") == 5000000000); }
    SECTION("convert #") { CHECK(parse_string_to_size("50") == 50); }
    SECTION("convert 0") { CHECK(parse_string_to_size("000") == 0); }
    SECTION("convert unexpected size character") { CHECK_THROWS(parse_string_to_size("5L")); }
    SECTION("convert not a number") { CHECK_THROWS(parse_string_to_size("abcd")); }
}

TEST_CASE("CliUtils: Extract tokens from dorado cmdline", TEST_GROUP) {
    std::string cmdline = "dorado basecaller model_path dataset --option1 blah";
    std::vector<std::string> expected_tokens = {"dorado",  "basecaller", "model_path",
                                                "dataset", "--option1",  "blah"};
    auto tokens = extract_token_from_cli(cmdline);
    CHECK(tokens.size() == 6);
    for (int i = 0; i < tokens.size(); i++) {
        CHECK(tokens[i] == expected_tokens[i]);
    }
}
