#include "models/kits.h"
#include "models/models.h"

#include <catch2/catch.hpp>

#include <stdexcept>
#include <tuple>

#define TEST_TAG "[ModelUtils]"

TEST_CASE(TEST_TAG " Get model sample rate by name") {
    SECTION("Check against valid 5khz model name") {
        CHECK(dorado::models::get_sample_rate_by_model_name(
                      "dna_r10.4.1_e8.2_400bps_fast@v4.2.0") == 5000);
    }
    SECTION("Check against valid 4khz model name") {
        CHECK(dorado::models::get_sample_rate_by_model_name(
                      "dna_r10.4.1_e8.2_260bps_fast@v4.0.0") == 4000);
    }
    SECTION("Check against unknown model name") {
        CHECK_THROWS_AS(dorado::models::get_sample_rate_by_model_name("blah"), std::runtime_error);
    }

    SECTION("Spot checks") {
        auto [model_name, sampling_rate] =
                GENERATE(table<std::string, dorado::models::SamplingRate>({
                        std::make_tuple("dna_r9.4.1_e8_fast@v3.4", 4000),
                        std::make_tuple("dna_r9.4.1_e8_hac@v3.3", 4000),
                        std::make_tuple("dna_r9.4.1_e8_sup@v3.3", 4000),
                        std::make_tuple("dna_r9.4.1_e8_sup@v3.6", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_fast@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_sup@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v3.5.2", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_fast@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_sup@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.0.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_fast@v4.1.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v4.1.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_260bps_sup@v4.1.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.1.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.1.0", 4000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.1.0", 4000),
                        // v4.2.0+ 5Khz
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.2.0", 5000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.2.0", 5000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.2.0", 5000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.3.0", 5000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.3.0", 5000),
                        std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.3.0", 5000),
                        // RNA2 3Khz
                        std::make_tuple("rna002_70bps_fast@v3", 3000),
                        std::make_tuple("rna002_70bps_hac@v3", 3000),
                        // RNA4 4Khz
                        std::make_tuple("rna004_130bps_fast@v3.0.1", 4000),
                        std::make_tuple("rna004_130bps_hac@v3.0.1", 4000),
                        std::make_tuple("rna004_130bps_sup@v3.0.1", 4000),
                }));

        CAPTURE(model_name);
        CAPTURE(sampling_rate);
        const auto result = dorado::models::get_sample_rate_by_model_name(model_name);
        CHECK(sampling_rate == result);
    }
}

TEST_CASE(TEST_TAG " Get simplex model info by name") {
    SECTION("Check all configured models") {
        for (const auto& model_name : dorado::models::simplex_model_names()) {
            const auto model_info = dorado::models::get_simplex_model_info(model_name);
            CAPTURE(model_name);
            CAPTURE(model_info.name);
            CHECK(model_name == model_info.name);
        }
    }

    SECTION("Spot checks") {
        auto [model_name] = GENERATE(table<std::string>({
                "dna_r9.4.1_e8_fast@v3.4",
                "dna_r9.4.1_e8_hac@v3.3",
                "dna_r9.4.1_e8_sup@v3.3",
                "dna_r9.4.1_e8_sup@v3.6",
                "dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
                "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
                "dna_r10.4.1_e8.2_260bps_sup@v3.5.2",
                "dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
                "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
                "dna_r10.4.1_e8.2_400bps_sup@v3.5.2",
                "dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
                "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
                "dna_r10.4.1_e8.2_260bps_sup@v4.0.0",
                "dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
                "dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
                "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
                "dna_r10.4.1_e8.2_260bps_sup@v4.1.0",
                "dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",
                "dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_fast@v4.3.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
                "rna002_70bps_fast@v3",
                "rna002_70bps_hac@v3",
                "rna004_130bps_fast@v3.0.1",
                "rna004_130bps_hac@v3.0.1",
                "rna004_130bps_sup@v3.0.1",
        }));

        CAPTURE(model_name);
        const auto model_info = dorado::models::get_simplex_model_info(model_name);
        CAPTURE(model_info.name);
        CHECK(model_name == model_info.name);
    }

    SECTION("Check unknown model raises") {
        CHECK_THROWS_AS(dorado::models::get_simplex_model_info("unknown"), std::runtime_error);
    }
}
