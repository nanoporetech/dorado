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
