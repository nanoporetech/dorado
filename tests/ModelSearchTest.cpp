#include "models/kits.h"
#include "models/metadata.h"
#include "models/model_complex.h"
#include "models/models.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <stdexcept>

#define TEST_TAG "[ModelComplexSearch]"

using namespace dorado::models;
namespace fs = std::filesystem;

using MC = ModelComplex;
using MCS = ModelComplexSearch;
using MVP = ModelVariantPair;
using MV = ModelVariant;
using ModsV = ModsVariant;
using ModsVP = ModsVariantPair;
using VV = ModelVersion;
using CC = Chemistry;

TEST_CASE(TEST_TAG "  ModelComplexSearch get_simplex_model_name", TEST_TAG) {
    SECTION("get_simplex_model_name all") {
        // given the model definitions the same model can be found
        for (const auto& mi : simplex_models()) {
            const auto complex =
                    to_string(mi.simplex.variant).append("@").append(to_string(mi.simplex.ver));
            const auto mf = MCS{MC{complex, mi.simplex}, mi.chemistry, false};
            CAPTURE(mi.name);
            CAPTURE(complex);
            CHECK(mf.simplex().name == mi.name);
        }
    }

    SECTION("get_simplex_model_name simplex spot checks") {
        // Check given the model definitions the same model can be found
        auto [chemistry, mvp, expected] = GENERATE(table<CC, MVP, std::string>({
                std::make_tuple(CC::DNA_R9_4_1_E8, MVP{MV::FAST, VV::v3_4_0},
                                "dna_r9.4.1_e8_fast@v3.4"),
                std::make_tuple(CC::DNA_R9_4_1_E8, MVP{MV::SUP, VV::v3_6_0},
                                "dna_r9.4.1_e8_sup@v3.6"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_260BPS, MVP{MV::HAC, VV::v3_5_2},
                                "dna_r10.4.1_e8.2_260bps_hac@v3.5.2"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::SUP, VV::v3_5_2},
                                "dna_r10.4.1_e8.2_400bps_sup@v3.5.2"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_260BPS, MVP{MV::FAST, VV::v4_0_0},
                                "dna_r10.4.1_e8.2_260bps_fast@v4.0.0"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::HAC, VV::v4_0_0},
                                "dna_r10.4.1_e8.2_400bps_hac@v4.0.0"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_260BPS, MVP{MV::SUP, VV::v4_1_0},
                                "dna_r10.4.1_e8.2_260bps_sup@v4.1.0"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::SUP, VV::v4_1_0},
                                "dna_r10.4.1_e8.2_400bps_sup@v4.1.0"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::SUP, VV::v4_2_0},
                                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0"),
                std::make_tuple(CC::DNA_R10_4_1_E8_2_APK_5KHZ, MVP{MV::SUP, VV::v5_0_0},
                                "dna_r10.4.1_e8.2_apk_sup@v5.0.0"),
                std::make_tuple(CC::RNA002_70BPS, MVP{MV::HAC, VV::v3_0_0}, "rna002_70bps_hac@v3"),
                std::make_tuple(CC::RNA004_130BPS, MVP{MV::HAC, VV::v3_0_1},
                                "rna004_130bps_hac@v3.0.1"),
        }));

        CAPTURE(expected);
        CAPTURE(to_string(chemistry));
        const auto variant = to_string(mvp.variant);
        const auto ver = to_string(mvp.ver);
        const auto complex = variant + "@" + ver;
        const auto mf = MCS{MC{complex, mvp}, chemistry, false};
        CAPTURE(complex);
        CHECK(mf.simplex().name == expected);
    }
}

TEST_CASE(TEST_TAG "  ModelComplexSearch get_stereo_model_name", TEST_TAG) {
    SECTION("get_stereo_model_name all") {
        // Check given the model definitions the same model can be found
        auto [chemistry, mvp, expected_simplex, expected_stereo] =
                GENERATE(table<CC, MVP, std::string, std::string>({
                        // 4kHz
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::HAC, VV::v4_0_0},
                                        "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
                                        "dna_r10.4.1_e8.2_4khz_stereo@v1.1"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::SUP, VV::v4_0_0},
                                        "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
                                        "dna_r10.4.1_e8.2_4khz_stereo@v1.1"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::HAC, VV::v4_1_0},
                                        "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
                                        "dna_r10.4.1_e8.2_4khz_stereo@v1.1"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_4KHZ, MVP{MV::SUP, VV::v4_1_0},
                                        "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",
                                        "dna_r10.4.1_e8.2_4khz_stereo@v1.1"),
                        // 5kHz
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::HAC, VV::v4_2_0},
                                        "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.1"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::SUP, VV::v4_2_0},
                                        "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.1"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::HAC, VV::v4_3_0},
                                        "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.2"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::SUP, VV::v4_3_0},
                                        "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.2"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::HAC, VV::v5_0_0},
                                        "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.3"),
                        std::make_tuple(CC::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::SUP, VV::v5_0_0},
                                        "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
                                        "dna_r10.4.1_e8.2_5khz_stereo@v1.3"),
                }));

        CAPTURE(expected_simplex);
        CAPTURE(expected_stereo);
        CAPTURE(to_string(chemistry));
        const auto variant = to_string(mvp.variant);
        const auto ver = to_string(mvp.ver);
        const auto complex = variant + "@" + ver;
        const auto mf = MCS{MC{complex, mvp}, chemistry, false};
        CAPTURE(complex);
        CHECK(mf.simplex().name == expected_simplex);
        CHECK(mf.stereo().name == expected_stereo);
    }
}

TEST_CASE(TEST_TAG "  ModelComplexSearch ModelComplexParser ", TEST_TAG) {
    // const auto foo = MS{}
    SECTION("ModelComplexParser parse expected") {
        auto [input, expected] = GENERATE(table<std::string, MC>({
                // No version
                std::make_tuple("auto", MC{"auto", MVP{MV::AUTO}}),
                std::make_tuple("fast", MC{"fast", MVP{MV::FAST}}),
                std::make_tuple("hac", MC{"hac", MVP{MV::HAC}}),
                std::make_tuple("sup", MC{"sup", MVP{MV::SUP}}),

                // specific version
                std::make_tuple("auto@v4.2.0", MC{"auto@v4.2.0", MVP{MV::AUTO, VV::v4_2_0}}),
                std::make_tuple("fast@v4.0.0", MC{"fast@v4.0.0", MVP{MV::FAST, VV::v4_0_0}}),
                std::make_tuple("hac@v4.2.0", MC{"hac@v4.2.0", MVP{MV::HAC, VV::v4_2_0}}),
                std::make_tuple("sup@v4.1.0", MC{"sup@v4.1.0", MVP{MV::SUP, VV::v4_1_0}}),

                // latest version
                std::make_tuple("auto@latest", MC{"autolatest", MVP{MV::AUTO}}),
                std::make_tuple("fast@latest", MC{"fastlatest", MVP{MV::FAST}}),
                std::make_tuple("hac@latest", MC{"hac@latest", MVP{MV::HAC}}),
                std::make_tuple("sup@latest", MC{"sup@latest", MVP{MV::SUP}}),

                // with single mods
                std::make_tuple("auto,5mC", MC{"auto,5mC", MVP{MV::AUTO}, {ModsVP{ModsV::M_5mC}}}),
                std::make_tuple("hac,4mC_5mC",
                                MC{"hac,4mC_5mC", MVP{MV::HAC}, {ModsVP{ModsV::M_4mC_5mC}}}),
                std::make_tuple("fast,5mC_5hmC",
                                MC{"fast,5mC_5hmC", MVP{MV::FAST}, {ModsVP{ModsV::M_5mC_5hmC}}}),
                std::make_tuple("auto,5mCG",
                                MC{"auto,5mCG", MVP{MV::AUTO}, {ModsVP{ModsV::M_5mCG}}}),
                std::make_tuple("hac,5mCG_5hmCG",
                                MC{"hac,5mCG_5hmCG", MVP{MV::HAC}, {ModsVP{ModsV::M_5mCG_5hmCG}}}),

                std::make_tuple("auto,6mA", MC{"auto,6mA", MVP{MV::AUTO}, {ModsVP{ModsV::M_6mA}}}),
                std::make_tuple("auto,m6A_DRACH",
                                MC{"auto,m6A_DRACH", MVP{MV::AUTO}, {ModsVP{ModsV::M_m6A_DRACH}}}),
                std::make_tuple("auto,m6A", MC{"auto,m6A", MVP{MV::AUTO}, {ModsVP{ModsV::M_m6A}}}),
                std::make_tuple("sup,pseU", MC{"sup,pseU", MVP{MV::SUP}, {ModsVP{ModsV::M_pseU}}}),
                std::make_tuple("sup,pseU,m6A", MC{"sup,pseU,m6A",
                                                   MVP{MV::SUP},
                                                   {ModsVP{ModsV::M_pseU}, ModsVP{ModsV::M_m6A}}}),
                // with single mods and version
                std::make_tuple("sup@v4.1.0,5mC@v2", MC{"sup@v4.1.0,5mC@v2",
                                                        MVP{MV::SUP, VV::v4_1_0},
                                                        {ModsVP{ModsV::M_5mC, VV::v2_0_0}}}),
                std::make_tuple("fast@latest,5mC_5hmC@v4.0.0",
                                MC{"fast@latest,5mC_5hmC@v4.0.0",
                                   MVP{MV::FAST},
                                   {ModsVP{ModsV::M_5mC_5hmC, VV::v4_0_0}}}),

                // Multi-mods
                std::make_tuple("auto,5mC,6mA", MC{"auto,5mC,6mA",
                                                   MVP{MV::AUTO},
                                                   {ModsVP{ModsV::M_5mC}, ModsVP{ModsV::M_6mA}}}),

                std::make_tuple("fast@latest,m6A_DRACH@v1,5mC_5hmC@v4.0.0",
                                MC{"fast@latest,m6A_DRACH@v1,5mC_5hmC@v4.0.0",
                                   MVP{MV::FAST},
                                   {ModsVP{ModsV::M_m6A_DRACH, VV::v1_0_0},
                                    ModsVP{ModsV::M_5mC_5hmC, VV::v4_0_0}}}),
        }));

        CAPTURE(input);
        auto result = ModelComplexParser::parse(input);
        CAPTURE(to_string(result.model.variant));
        CAPTURE(to_string(result.model.ver));
        CAPTURE(result.mods.size());
        CHECK(result.model.variant == expected.model.variant);

        CHECK(result.model.ver == expected.model.ver);
        CHECK(result.mods.size() == expected.mods.size());

        CHECK_FALSE(result.is_path());
        CHECK(result.has_model_variant());

        for (size_t i = 0; i < result.mods.size(); ++i) {
            const auto& res = result.mods.at(i);
            const auto& ex = expected.mods.at(i);
            CHECK(res.variant == ex.variant);
            CHECK(res.ver == ex.ver);
            CHECK(result.has_mods_variant());
        }
    }

    SECTION("ModelComplexParser parse expected path") {
        auto [input] = GENERATE(table<std::string>({
                // No version
                std::make_tuple("dna_r10.4.1_e8.2_260bps@4.2.0"),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_4khz@4.2.0"),
                std::make_tuple("dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2"),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC_5hmC@v1"),
                std::make_tuple("rna004_130bps_sup@v3.0.1_m6A_DRACH@v1"),
                std::make_tuple("hac/dna_r10.4.1_e8.2_400bps_5khz@4.2.0"),
                std::make_tuple("../auto/fast/dna_r9.4.1_e8@3.5.0"),
                std::make_tuple("~/sup/rna002_70bps@4.1.0"),
                std::make_tuple("rna004_130bps@4.2.0"),
                std::make_tuple("foo"),
                std::make_tuple("sup/foo/"),
                std::make_tuple("./auto"),
                std::make_tuple("./fast"),
                std::make_tuple("./hac"),
                std::make_tuple("./sup"),
        }));

        CAPTURE(input);
        auto result = ModelComplexParser::parse(input);
        CHECK(result.raw == input);
        CHECK(result.model.variant == ModelVariant::NONE);
        CHECK(result.model.ver == ModelVersion::NONE);
        CHECK(result.mods.size() == 0);
        CHECK(result.is_path());
        CHECK_FALSE(result.has_model_variant());
        CHECK_FALSE(result.has_mods_variant());
    }

    SECTION("ModelComplexParser parse_version expected") {
        // If a model version is parsed ok, but is not a recognised version then a different
        // error is raised explaining this

        // clang-format off
    auto [input, expected] = GENERATE(
        table<std::string, std::string>({
             std::make_tuple("v1.2.3", "v1.2.3"),
             std::make_tuple("v0.0.0", "v0.0.0"),
             std::make_tuple("v12.345.678", "v12.345.678"),
             std::make_tuple("v12.34.56.78", "v12.34.56.78"),
             std::make_tuple("v0", "v0.0.0"),
             std::make_tuple("v1.", "v1.0.0"),
             std::make_tuple("v2..", "v2.0.0"),
             std::make_tuple("v.", "v0.0.0"),
             std::make_tuple("v...", "v0.0.0.0"),
             std::make_tuple("v4.1.", "v4.1.0"),
             std::make_tuple("V0.", "v0.0.0"),
        }));
        // clang-format on

        CAPTURE(input);
        auto result = ModelComplexParser::parse_version(input);
        CHECK(result == expected);
    }

    SECTION("ModelComplexParser parse_version unexpected values") {
        auto [input] = GENERATE(table<std::string>({
                "",
                "n",
                "v1.-2.3",
                "v-1.-2.-3",
                "va",
                "v1.1a",
                "v1. 1a",
                "v. 1a . t",
        }));
        CAPTURE(input);
        CHECK_THROWS_AS(ModelComplexParser::parse_version(input), std::runtime_error);
    }
}
