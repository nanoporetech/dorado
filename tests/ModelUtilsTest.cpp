#include "TestUtils.h"
#include "models/kits.h"
#include "models/models.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <set>
#include <stdexcept>
#include <tuple>

#define TEST_TAG "[ModelUtils]"

CATCH_TEST_CASE(TEST_TAG " Get model sample rate by name") {
    CATCH_SECTION("Check against valid 5khz model name") {
        CATCH_CHECK(dorado::models::get_sample_rate_by_model_name(
                            "dna_r10.4.1_e8.2_400bps_fast@v4.2.0") == 5000);
    }
    CATCH_SECTION("Check against unknown model name") {
        CATCH_CHECK_THROWS_AS(dorado::models::get_sample_rate_by_model_name("blah"),
                              std::runtime_error);
    }

    CATCH_SECTION("Spot checks") {
        auto [model_name, sampling_rate] = GENERATE(table<std::string, int>({
                // v4.2.0+ 5Khz
                std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.2.0", 5000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.2.0", 5000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.2.0", 5000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_fast@v4.3.0", 5000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.3.0", 5000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.3.0", 5000),
                // RNA4 4Khz
                std::make_tuple("rna004_130bps_fast@v3.0.1", 4000),
                std::make_tuple("rna004_130bps_hac@v3.0.1", 4000),
                std::make_tuple("rna004_130bps_sup@v3.0.1", 4000),
        }));

        CATCH_CAPTURE(model_name);
        CATCH_CAPTURE(sampling_rate);
        const auto result = dorado::models::get_sample_rate_by_model_name(model_name);
        CATCH_CHECK(static_cast<dorado::models::SamplingRate>(sampling_rate) == result);
    }

    CATCH_SECTION("Check deprecated models") {
        auto [model_name, sampling_rate] = GENERATE(table<std::string, int>({
                // DNA R9.4.1
                std::make_tuple("dna_r9.4.1_e8_fast@v3.4", 4000),
                std::make_tuple("dna_r9.4.1_e8_hac@v3.3", 4000),
                std::make_tuple("dna_r9.4.1_e8_sup@v3.3", 4000),
                std::make_tuple("dna_r9.4.1_e8_sup@v3.6", 4000),

                // DNA r10.4.1 - 4kHz
                std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v3.5.2", 4000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v3.5.2", 4000),
                std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v4.0.0", 4000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.0.0", 4000),
                std::make_tuple("dna_r10.4.1_e8.2_260bps_hac@v4.1.0", 4000),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_hac@v4.1.0", 4000),

                // RNA2 3Khz
                std::make_tuple("rna002_70bps_fast@v3", 3000),
                std::make_tuple("rna002_70bps_hac@v3", 3000),
        }));

        CATCH_CAPTURE(model_name);
        CATCH_CAPTURE(sampling_rate);
        CATCH_CHECK_THROWS_AS(dorado::models::get_sample_rate_by_model_name(model_name),
                              std::runtime_error);
    }
}

CATCH_TEST_CASE(TEST_TAG " Get simplex model info by name") {
    CATCH_SECTION("Check all configured models") {
        for (const auto& model_name : dorado::models::simplex_model_names()) {
            const auto model_info = dorado::models::get_simplex_model_info(model_name);
            CATCH_CAPTURE(model_name);
            CATCH_CAPTURE(model_info.name);
            CATCH_CHECK(model_name == model_info.name);
        }
    }

    CATCH_SECTION("Spot checks") {
        auto [model_name] = GENERATE(table<std::string>({
                "dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
                "dna_r10.4.1_e8.2_400bps_fast@v4.3.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
                "rna004_130bps_fast@v3.0.1",
                "rna004_130bps_hac@v3.0.1",
                "rna004_130bps_sup@v3.0.1",
                "rna004_130bps_sup@v5.1.0",
        }));

        CATCH_CAPTURE(model_name);
        const auto model_info = dorado::models::get_simplex_model_info(model_name);
        CATCH_CAPTURE(model_info.name);
        CATCH_CHECK(model_name == model_info.name);
    }

    CATCH_SECTION("Throws on deprecated model") {
        auto [model_name] = GENERATE(table<std::string>({
                "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
                "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
                "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
                "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
                "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
        }));

        CATCH_CAPTURE(model_name);
        CATCH_CHECK_THROWS_AS(dorado::models::get_simplex_model_info(model_name),
                              std::runtime_error);
    }

    CATCH_SECTION("Check unknown model raises") {
        CATCH_CHECK_THROWS_AS(dorado::models::get_simplex_model_info("unknown"),
                              std::runtime_error);
    }

    CATCH_SECTION("Check deprecated model throws") {
        auto [model_name] = GENERATE(table<std::string>({
                "dna_r9.4.1_e8_fast@v3.4",
                "dna_r9.4.1_e8_hac@v3.3",
                "dna_r9.4.1_e8_sup@v3.3",
                "dna_r9.4.1_e8_sup@v3.6",
                "rna002_70bps_fast@v3",
                "rna002_70bps_hac@v3",
        }));

        CATCH_CAPTURE(model_name);
        CATCH_CHECK_THROWS_AS(dorado::models::get_simplex_model_info(model_name),
                              std::runtime_error);
    }

    CATCH_SECTION("Check all models unique") {
        using namespace dorado::models;
        std::set<std::string> all_models;
        for (const ModelList& models :
             {simplex_models(), simplex_deprecated_models(), stereo_models(), modified_models(),
              modified_deprecated_models(), correction_models(), polish_models(),
              variant_models()}) {
            for (const ModelInfo& model : models) {
                CATCH_CAPTURE(model.name);
                auto [_, was_emplaced] = all_models.emplace(model.name);
                CATCH_CHECK(was_emplaced);
            }
        }
    }
}

CATCH_TEST_CASE(TEST_TAG "  get_supported_model_info", TEST_TAG) {
    std::vector<std::string> expected_models = {
            "rna004_130bps_fast@v3.0.1",
            "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
            "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1",
            "dna_r10.4.1_e8.2_5khz_stereo@v1.2",
    };
    CATCH_SECTION("No path") {
        std::string model_info = dorado::models::get_supported_model_info("");

        // Check that it seems to contain some models we expect
        for (const auto& expected_model : expected_models) {
            CATCH_CAPTURE(expected_model);
            CATCH_CHECK(model_info.find(expected_model) != std::string::npos);
        }
    }

    CATCH_SECTION("Path Filtering") {
        auto tmp_dir = make_temp_dir("get_supported_model_info_test");

        // This should return no models as they don't exist in the directory
        std::string model_info = dorado::models::get_supported_model_info(tmp_dir.m_path.string());
        for (const auto& expected_model : expected_models) {
            CATCH_CHECK(model_info.find(expected_model) == std::string::npos);
        }

        // Making the modbase dir should not make it appear, as it's canonical model doesn't exist
        std::filesystem::create_directory(tmp_dir.m_path /
                                          "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1");
        model_info = dorado::models::get_supported_model_info(tmp_dir.m_path.string());
        CATCH_CHECK(model_info.find("dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1") ==
                    std::string::npos);

        // Adding the canonical model dir should make it appear, and the modbase model above, but not the stereo model
        std::filesystem::create_directory(tmp_dir.m_path / "dna_r10.4.1_e8.2_400bps_sup@v4.3.0");
        model_info = dorado::models::get_supported_model_info(tmp_dir.m_path.string());
        CATCH_CHECK(model_info.find("dna_r10.4.1_e8.2_400bps_sup@v4.3.0") != std::string::npos);
        CATCH_CHECK(model_info.find("dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1") !=
                    std::string::npos);
        CATCH_CHECK(model_info.find("dna_r10.4.1_e8.2_5khz_stereo@v1.2") == std::string::npos);

        // Adding the stereo model dir should make it appear.
        std::filesystem::create_directory(tmp_dir.m_path / "dna_r10.4.1_e8.2_5khz_stereo@v1.2");
        model_info = dorado::models::get_supported_model_info(tmp_dir.m_path.string());
        CATCH_CHECK(model_info.find("dna_r10.4.1_e8.2_5khz_stereo@v1.2") != std::string::npos);
    }
}