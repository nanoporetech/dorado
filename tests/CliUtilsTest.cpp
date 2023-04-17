#include "utils/cli_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[cli_utils]"

using namespace dorado::utils;

TEST_CASE("CliUtils: Check thread allocation", TEST_GROUP) {
    int aligner_threads, writer_threads;

    SECTION("aligner threads 0 and writer threads >0") {
        std::tie(aligner_threads, writer_threads) =
                aligner_writer_thread_allocation(0, 5, 10, 0.25f);
        REQUIRE(aligner_threads == 5);
        REQUIRE(writer_threads == 5);
    }

    SECTION("aligner threads >0 and writer threads 0") {
        std::tie(aligner_threads, writer_threads) =
                aligner_writer_thread_allocation(5, 0, 10, 0.25f);
        REQUIRE(aligner_threads == 5);
        REQUIRE(writer_threads == 5);
    }

    SECTION("aligner threads >0 and writer threads >0") {
        std::tie(aligner_threads, writer_threads) =
                aligner_writer_thread_allocation(4, 4, 10, 0.25f);
        REQUIRE(aligner_threads == 4);
        REQUIRE(writer_threads == 4);
    }

    SECTION("aligner threads 0 and writer threads 0") {
        std::tie(aligner_threads, writer_threads) =
                aligner_writer_thread_allocation(0, 0, 10, 0.25f);
        REQUIRE(aligner_threads == 3);
        REQUIRE(writer_threads == 7);
    }
}
