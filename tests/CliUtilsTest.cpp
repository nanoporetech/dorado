#include "TestUtils.h"
#include "cli/cli_utils.h"
#include "cli/model_resolution.h"

// Catch must come last so we can undo torch defining CHECK.
#undef CHECK
#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[cli_utils]"

using namespace dorado::cli;
using namespace dorado::model_resolution;

TEST_CASE("CliUtils: Check thread allocation", TEST_GROUP) {
    int aligner_threads, writer_threads;

    SECTION("fraction > 0") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 0.25f);
        REQUIRE(aligner_threads == 8);
        REQUIRE(writer_threads == 2);
    }

    SECTION("fraction == 0") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 0.0f);
        REQUIRE(aligner_threads == 9);
        REQUIRE(writer_threads == 1);
    }

    SECTION("fraction 100%") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 1.0f);
        REQUIRE(aligner_threads == 1);
        REQUIRE(writer_threads == 9);
    }
}

TEST_CASE("CliUtils: Extract tokens from dorado cmdline", TEST_GROUP) {
    std::string cmdline = "dorado basecaller model_path dataset --option1 blah";
    std::vector<std::string> expected_tokens = {"dorado",  "basecaller", "model_path",
                                                "dataset", "--option1",  "blah"};
    auto tokens = extract_token_from_cli(cmdline);
    CHECK(tokens.size() == 6);
    for (size_t i = 0; i < tokens.size(); i++) {
        CHECK(tokens[i] == expected_tokens[i]);
    }
}
