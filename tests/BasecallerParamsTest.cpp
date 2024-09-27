#include "basecall/BasecallerParams.h"

#include "TestUtils.h"
#include "basecall/CRFModelConfig.h"
#include "utils/parameters.h"

#include <catch2/catch.hpp>

#include <optional>

#define CUT_TAG "[BasecallerParams]"

namespace fs = std::filesystem;
using dorado::basecall::BasecallerParams;
using dorado::utils::default_parameters;

TEST_CASE(CUT_TAG ": test default constructor", CUT_TAG) {
    const auto base = BasecallerParams{};
    CHECK(base.chunk_size() == default_parameters.chunksize);
    CHECK(base.overlap() == default_parameters.overlap);
    CHECK(base.batch_size() == default_parameters.batchsize);
}

TEST_CASE(CUT_TAG ": test update from config", CUT_TAG) {
    auto base = BasecallerParams{};
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
    base.update(path);

    CHECK(base.chunk_size() == 12288);
    CHECK(base.overlap() == 600);
    // batchsize is ignored in the config
    CHECK(base.batch_size() == default_parameters.batchsize);
}

TEST_CASE(CUT_TAG ": test update from config no overwrite CLI", CUT_TAG) {
    auto base = BasecallerParams{};

    base.update(BasecallerParams::Priority::CLI_ARG, 1, std::nullopt, 3);

    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
    base.update(path);

    CHECK(base.chunk_size() == 1);
    CHECK(base.overlap() == 600);
    CHECK(base.batch_size() == 3);
}

TEST_CASE(CUT_TAG ": test update from CLI", CUT_TAG) {
    auto base = BasecallerParams{};

    const int cs = 1234;
    const int ov = 432;
    const int bs = 61;
    base.update(BasecallerParams::Priority::CLI_ARG, cs, ov, bs);
    CHECK(base.chunk_size() == cs);
    CHECK(base.overlap() == ov);
    CHECK(base.batch_size() == bs);
}

TEST_CASE(CUT_TAG ": test update from CLI optional", CUT_TAG) {
    auto base = BasecallerParams{};

    const int cs = 234;
    const int bs = 90;
    base.update(BasecallerParams::Priority::CLI_ARG, cs, std::nullopt, bs);
    CHECK(base.chunk_size() == cs);
    CHECK(base.batch_size() == bs);

    // Checking that the default is unchanged for nullopt
    CHECK(base.overlap() == default_parameters.overlap);
}

TEST_CASE(CUT_TAG ": test priority", CUT_TAG) {
    SECTION("default no overwrite default") {
        auto base = BasecallerParams{};
        base.update(BasecallerParams::Priority::DEFAULT, 1, 2, 3);
        // Assert values not updated as new priority is not greater than existing
        CHECK(base.chunk_size() != 1);
        CHECK(base.overlap() != 2);
        CHECK(base.batch_size() != 3);
        CHECK(base.chunk_size() == default_parameters.chunksize);
        CHECK(base.overlap() == default_parameters.overlap);
        CHECK(base.batch_size() == default_parameters.batchsize);
    }

    SECTION("default < config < cli < force") {
        auto base = BasecallerParams{};

        // Apply in descending order demonstrates that lesser priority cannot overwrite greater
        base.update(BasecallerParams::Priority::FORCE, std::nullopt, std::nullopt, 333);
        base.update(BasecallerParams::Priority::CLI_ARG, std::nullopt, 22, 33);
        base.update(BasecallerParams::Priority::CONFIG, 1, 2, 3);

        // Assert values not updated as new priority is not greater than existing
        CHECK(base.chunk_size() == 1);
        CHECK(base.overlap() == 22);
        CHECK(base.batch_size() == 333);
    }

    SECTION("equal priority (not force) no update") {
        auto base = BasecallerParams{};

        base.update(BasecallerParams::Priority::CLI_ARG, 11, 22, 33);
        base.update(BasecallerParams::Priority::CLI_ARG, 999, 999, 999);

        CHECK(base.chunk_size() == 11);
        CHECK(base.overlap() == 22);
        CHECK(base.batch_size() == 33);
    }

    SECTION("force always update") {
        auto base = BasecallerParams{};

        base.update(BasecallerParams::Priority::FORCE, 111, 222, 333);
        base.update(BasecallerParams::Priority::FORCE, 19, 29, 39);

        CHECK(base.chunk_size() == 19);
        CHECK(base.overlap() == 29);
        CHECK(base.batch_size() == 39);
    }

    SECTION("merge uses priority") {
        auto base = BasecallerParams{};
        base.update(BasecallerParams::Priority::CONFIG, std::nullopt, 2, 3);

        auto other = BasecallerParams{};
        other.update(BasecallerParams::Priority::CLI_ARG, std::nullopt, std::nullopt, 33);

        base.update(other);

        CHECK(base.chunk_size() == default_parameters.chunksize);
        CHECK(base.overlap() == 2);
        CHECK(base.batch_size() == 33);
    }

    SECTION("setters use force") {
        auto base = BasecallerParams{};

        base.update(BasecallerParams::Priority::FORCE, 1, 2, 3);
        base.set_chunk_size(111);
        base.set_overlap(222);
        base.set_batch_size(333);

        CHECK(base.chunk_size() == 111);
        CHECK(base.overlap() == 222);
        CHECK(base.batch_size() == 333);
    }
}

TEST_CASE(CUT_TAG ": test assertions", CUT_TAG) {
    SECTION("setting negative values throw") {
        auto base = BasecallerParams{};
        const auto msg = "BasecallerParams::set_value value must be positive integer";
        CHECK_THROWS_WITH(base.set_chunk_size(-1), msg);
        CHECK_THROWS_WITH(base.set_overlap(-2), msg);
        CHECK_THROWS_WITH(base.set_batch_size(-1), msg);
    }
}
