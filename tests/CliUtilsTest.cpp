#include "TestUtils.h"
#include "cli/cli_utils.h"
#include "cli/model_resolution.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[cli_utils]"

using namespace dorado::cli;
using namespace dorado::model_resolution;

CATCH_TEST_CASE("CliUtils: Check thread allocation", TEST_GROUP) {
    int aligner_threads, writer_threads;

    CATCH_SECTION("fraction > 0") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 0.25f);
        CATCH_REQUIRE(aligner_threads == 8);
        CATCH_REQUIRE(writer_threads == 2);
    }

    CATCH_SECTION("fraction == 0") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 0.0f);
        CATCH_REQUIRE(aligner_threads == 9);
        CATCH_REQUIRE(writer_threads == 1);
    }

    CATCH_SECTION("fraction 100%") {
        std::tie(aligner_threads, writer_threads) = worker_vs_writer_thread_allocation(10, 1.0f);
        CATCH_REQUIRE(aligner_threads == 1);
        CATCH_REQUIRE(writer_threads == 9);
    }
}

CATCH_TEST_CASE("CliUtils: Extract tokens from dorado cmdline", TEST_GROUP) {
    std::string cmdline = "dorado basecaller model_path dataset --option1 blah";
    std::vector<std::string> expected_tokens = {"dorado",  "basecaller", "model_path",
                                                "dataset", "--option1",  "blah"};
    auto tokens = extract_token_from_cli(cmdline);
    CATCH_CHECK(tokens.size() == 6);
    for (size_t i = 0; i < tokens.size(); i++) {
        CATCH_CHECK(tokens[i] == expected_tokens[i]);
    }
}
