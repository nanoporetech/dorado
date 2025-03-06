
#include "config/BatchParams.h"

#include "TestUtils.h"
#include "utils/parameters.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <optional>

#define CUT_TAG "[BatchParams]"

namespace fs = std::filesystem;
using dorado::config::BatchParams;
using dorado::utils::default_parameters;

CATCH_TEST_CASE(CUT_TAG ": test default constructor", CUT_TAG) {
    const auto base = BatchParams{};
    CATCH_CHECK(base.chunk_size() == default_parameters.chunksize);
    CATCH_CHECK(base.overlap() == default_parameters.overlap);
    CATCH_CHECK(base.batch_size() == default_parameters.batchsize);
}

CATCH_TEST_CASE(CUT_TAG ": test update from config", CUT_TAG) {
    auto base = BatchParams{};
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
    base.update(path);

    CATCH_CHECK(base.chunk_size() == 12288);
    CATCH_CHECK(base.overlap() == 600);
    // batchsize is ignored in the config
    CATCH_CHECK(base.batch_size() == default_parameters.batchsize);
}

CATCH_TEST_CASE(CUT_TAG ": test update from config no overwrite CLI", CUT_TAG) {
    auto base = BatchParams{};

    base.update(BatchParams::Priority::CLI_ARG, 1, std::nullopt, 3);

    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
    base.update(path);

    CATCH_CHECK(base.chunk_size() == 1);
    CATCH_CHECK(base.overlap() == 600);
    CATCH_CHECK(base.batch_size() == 3);
}

CATCH_TEST_CASE(CUT_TAG ": test update from CLI", CUT_TAG) {
    auto base = BatchParams{};

    const int cs = 1234;
    const int ov = 432;
    const int bs = 61;
    base.update(BatchParams::Priority::CLI_ARG, cs, ov, bs);
    CATCH_CHECK(base.chunk_size() == cs);
    CATCH_CHECK(base.overlap() == ov);
    CATCH_CHECK(base.batch_size() == bs);
}

CATCH_TEST_CASE(CUT_TAG ": test update from CLI optional", CUT_TAG) {
    auto base = BatchParams{};

    const int cs = 234;
    const int bs = 90;
    base.update(BatchParams::Priority::CLI_ARG, cs, std::nullopt, bs);
    CATCH_CHECK(base.chunk_size() == cs);
    CATCH_CHECK(base.batch_size() == bs);

    // Checking that the default is unchanged for nullopt
    CATCH_CHECK(base.overlap() == default_parameters.overlap);
}

CATCH_TEST_CASE(CUT_TAG ": test priority", CUT_TAG) {
    CATCH_SECTION("default no overwrite default") {
        auto base = BatchParams{};
        base.update(BatchParams::Priority::DEFAULT, 1, 2, 3);
        // Assert values not updated as new priority is not greater than existing
        CATCH_CHECK(base.chunk_size() != 1);
        CATCH_CHECK(base.overlap() != 2);
        CATCH_CHECK(base.batch_size() != 3);
        CATCH_CHECK(base.chunk_size() == default_parameters.chunksize);
        CATCH_CHECK(base.overlap() == default_parameters.overlap);
        CATCH_CHECK(base.batch_size() == default_parameters.batchsize);
    }

    CATCH_SECTION("default < config < cli < force") {
        auto base = BatchParams{};

        // Apply in descending order demonstrates that lesser priority cannot overwrite greater
        base.update(BatchParams::Priority::FORCE, std::nullopt, std::nullopt, 333);
        base.update(BatchParams::Priority::CLI_ARG, std::nullopt, 22, 33);
        base.update(BatchParams::Priority::CONFIG, 1, 2, 3);

        // Assert values not updated as new priority is not greater than existing
        CATCH_CHECK(base.chunk_size() == 1);
        CATCH_CHECK(base.overlap() == 22);
        CATCH_CHECK(base.batch_size() == 333);
    }

    CATCH_SECTION("equal priority (not force) no update") {
        auto base = BatchParams{};

        base.update(BatchParams::Priority::CLI_ARG, 11, 22, 33);
        base.update(BatchParams::Priority::CLI_ARG, 999, 999, 999);

        CATCH_CHECK(base.chunk_size() == 11);
        CATCH_CHECK(base.overlap() == 22);
        CATCH_CHECK(base.batch_size() == 33);
    }

    CATCH_SECTION("force always update") {
        auto base = BatchParams{};

        base.update(BatchParams::Priority::FORCE, 111, 222, 333);
        base.update(BatchParams::Priority::FORCE, 19, 29, 39);

        CATCH_CHECK(base.chunk_size() == 19);
        CATCH_CHECK(base.overlap() == 29);
        CATCH_CHECK(base.batch_size() == 39);
    }

    CATCH_SECTION("merge uses priority") {
        auto base = BatchParams{};
        base.update(BatchParams::Priority::CONFIG, std::nullopt, 2, 3);

        auto other = BatchParams{};
        other.update(BatchParams::Priority::CLI_ARG, std::nullopt, std::nullopt, 33);

        base.update(other);

        CATCH_CHECK(base.chunk_size() == default_parameters.chunksize);
        CATCH_CHECK(base.overlap() == 2);
        CATCH_CHECK(base.batch_size() == 33);
    }

    CATCH_SECTION("setters use force") {
        auto base = BatchParams{};

        base.update(BatchParams::Priority::FORCE, 1, 2, 3);
        base.set_chunk_size(111);
        base.set_overlap(222);
        base.set_batch_size(333);

        CATCH_CHECK(base.chunk_size() == 111);
        CATCH_CHECK(base.overlap() == 222);
        CATCH_CHECK(base.batch_size() == 333);
    }
}

CATCH_TEST_CASE(CUT_TAG ": test assertions", CUT_TAG) {
    CATCH_SECTION("setting negative values throw") {
        auto base = BatchParams{};
        const auto msg = "BatchParams::set_value value must be positive integer";
        CATCH_CHECK_THROWS_WITH(base.set_chunk_size(-1), msg);
        CATCH_CHECK_THROWS_WITH(base.set_overlap(-2), msg);
        CATCH_CHECK_THROWS_WITH(base.set_batch_size(-1), msg);
    }
}
