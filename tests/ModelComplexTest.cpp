#include "TestUtils.h"
#include "model_resolver/ModelResolver.h"
#include "models/kits.h"
#include "models/metadata.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "utils/fs_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iterator>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#define TEST_TAG "[ModelComplex]"

using namespace dorado::models;
using namespace dorado::model_resolution;

namespace fs = std::filesystem;

using MC = ModelComplex;
using MVP = ModelVariantPair;
using ModVP = ModsVariantPair;
using ModV = ModsVariant;
using MV = ModelVariant;
using VV = ModelVersion;

CATCH_TEST_CASE(TEST_TAG "  parse", TEST_TAG) {
    CATCH_SECTION(" parse all model simplex variants") {
        // Test we can parse all known simplex model variants
        for (const auto &simplex_info : simplex_models()) {
            const auto model_variant_string = to_string(simplex_info.simplex.variant) + "@" +
                                              to_string(simplex_info.simplex.ver);

            CATCH_CAPTURE(simplex_info.name);
            CATCH_CAPTURE(model_variant_string);

            ModelComplex parsed_complex = ModelComplex::parse(model_variant_string);
            CATCH_CHECK(parsed_complex.get_raw() == model_variant_string);
            CATCH_CHECK(parsed_complex.is_variant_style());
            const bool is_variant_equal =
                    parsed_complex.get_simplex_model_variant() == simplex_info.simplex;
            CATCH_CHECK(is_variant_equal);
            const bool is_mods_empty = parsed_complex.get_mod_model_variants().empty();
            CATCH_CHECK(is_mods_empty);
        }
    }

    CATCH_SECTION(" parse all simplex names") {
        // Test we can parse all known simplex and modbase model names
        for (const auto &simplex_info : simplex_models()) {
            CATCH_CAPTURE(simplex_info.name);

            ModelComplex parsed_complex = ModelComplex::parse(simplex_info.name);
            CATCH_CHECK(parsed_complex.get_raw() == simplex_info.name);
            CATCH_CHECK(parsed_complex.is_named_style());
            const bool is_name_equal = parsed_complex.get_named_simplex_model() == simplex_info;
            CATCH_CHECK(is_name_equal);
            const bool is_mods_empty = parsed_complex.get_named_mods_models().empty();
            CATCH_CHECK(is_mods_empty);
        }
    }

    CATCH_SECTION(" parse all modbase names") {
        // Test we can parse all known simplex and modbase model names
        for (const auto &modbase_info : modified_models()) {
            CATCH_CAPTURE(modbase_info.name);

            ModelComplex parsed_complex = ModelComplex::parse(modbase_info.name);
            CATCH_CHECK(parsed_complex.get_raw() == modbase_info.name);
            CATCH_CHECK(parsed_complex.is_named_style());
            CATCH_CHECK(parsed_complex.get_named_simplex_model() ==
                        get_modbase_model_simplex_parent(modbase_info));
            CATCH_CHECK(parsed_complex.get_named_mods_models().size() == 1);
            CATCH_CHECK(parsed_complex.get_named_mods_models().at(0) == modbase_info);
        }
    }

    CATCH_SECTION(" parse example variants") {
        // Test that we can reconstruct model variant and version strings
        auto [mvp] = GENERATE(table<ModelVariantPair>({
                MVP{MV::SUP, VV::v4_2_0},
                MVP{MV::SUP, VV::v5_0_0},
                MVP{MV::HAC, VV::v3_0_1},
                MVP{MV::FAST, VV::v5_2_0},
                MVP{MV::AUTO, VV::v1_0_0},
        }));

        const auto simplex_complex_str = to_string(mvp.variant) + "@" + to_string(mvp.ver);
        {
            CATCH_CAPTURE(simplex_complex_str);
            const auto mc = ModelComplex::parse(simplex_complex_str);
            CATCH_CHECK(mc.is_variant_style());
            CATCH_CHECK(mc.get_simplex_model_variant() == mvp);
        }

        auto [mod] = GENERATE(table<ModVP>({
                ModVP{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
                ModVP{ModsVariant::M_inosine_m6A, VV::v5_0_0},
                ModVP{ModsVariant::M_m6A, VV::v3_0_1},
                ModVP{ModsVariant::M_pseU_2OmeU, VV::v5_2_0},
                ModVP{ModsVariant::M_2OmeG, VV::v1_0_0},
        }));

        const auto modbase_complex_str =
                simplex_complex_str + "," + to_string(mod.variant) + "@" + to_string(mod.ver);
        {
            CATCH_CAPTURE(modbase_complex_str);
            const auto mc = ModelComplex::parse(modbase_complex_str);
            CATCH_CHECK(mc.is_variant_style());
            CATCH_CHECK(mc.get_simplex_model_variant() == mvp);
            CATCH_CHECK(mc.get_mod_model_variants().size() == 1);
            CATCH_CHECK(mc.get_mod_model_variants()[0] == mod);
        }
    }

    CATCH_SECTION(" parse more example variants") {
        auto [complex, simplex, mods] = GENERATE(table<std::string, MVP, std::vector<ModVP>>({
                {
                        "hac",
                        MVP{MV::HAC, VV::NONE},
                        {},
                },
                {
                        "fast@v1.0",
                        MVP{MV::FAST, VV::v1_0_0},
                        {},
                },
                {
                        "sup@v2,5mC_5hmC",
                        MVP{MV::SUP, VV::v2_0_0},
                        {ModVP{ModsVariant::M_5mC_5hmC, VV::NONE}},
                },
                {
                        "auto@v5.1.0,6mA,2OmeG@v1",
                        MVP{MV::AUTO, VV::v5_1_0},
                        {
                                ModVP{ModsVariant::M_6mA, VV::NONE},
                                ModVP{ModsVariant::M_2OmeG, VV::v1_0_0},
                        },
                },
                {
                        "fast@latest,m6A_DRACH@v1,5mC_5hmC@v4.0.0",
                        MVP{MV::FAST, VV::NONE},
                        {
                                ModVP{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
                                ModVP{ModsVariant::M_5mC_5hmC, VV::v4_0_0},
                        },
                },
        }));
        CATCH_CAPTURE(complex);
        const auto mc = ModelComplex::parse(complex);
        CATCH_CHECK(mc.get_simplex_model_variant() == simplex);
        CATCH_CHECK(mc.get_mod_model_variants() == mods);
    }

    CATCH_SECTION(" parse path like") {
        auto [input] = GENERATE(table<std::string>({
                "hac/dna_r10.4.1_e8.2_400bps_5khz@4.2.0",
                "../auto/fast/dna_r9.4.1_e8@3.5.0",
                "rna004_130bps@4.2.0",
                "foo",
                "sup/foo/",
                "./auto",
                "./fast",
                "./hac",
                "./sup",
        }));

        CATCH_CAPTURE(input);
        auto result = ModelComplex::parse(input);

        CATCH_CHECK(result.get_raw() == input);
        CATCH_CHECK(result.is_path_style());
    }

    CATCH_SECTION(" parse version strings") {
        class TestVersionParser final : public ModelComplex {
        public:
            static std::string parse_version(const std::string &version) {
                return ModelComplex::parse_variant_version(version);
            }
        };
        auto [input, expected] = GENERATE(table<std::string, std::string>({
                std::make_tuple("v1.2.3", "v1.2.3"),
                std::make_tuple("v0.0.0", "v0.0.0"),
                std::make_tuple("v12.345.678", "v12.345.678"),
                std::make_tuple("v12.34.56.78", "v12.34.56.78"),
                std::make_tuple("v0", "v0.0.0"),
                std::make_tuple("v4.1", "v4.1.0"),
        }));

        CATCH_CAPTURE(input);
        CATCH_CHECK(TestVersionParser::parse_version(input) == expected);
    }

    CATCH_SECTION(" catch invalid version strings") {
        class TestVersionParser final : public ModelComplex {
        public:
            static std::string parse_version(const std::string &version) {
                return ModelComplex::parse_variant_version(version);
            }
        };
        auto [input] = GENERATE(table<std::string>(
                {"", "v", "v.", "v..0", "v1.", "v2..", "v4.1.", "x0.0.0", "v0.x.0"}));

        CATCH_CAPTURE(input);
        CATCH_CHECK_THROWS_AS(TestVersionParser::parse_version(input), std::runtime_error);
    }
}

CATCH_TEST_CASE(TEST_TAG "  ModelComplexSearch ", TEST_TAG) {
    CATCH_SECTION("ModelComplexSearch parse variant ") {
        struct In {
            const std::string complex;
            const std::optional<Chemistry> chemistry{std::nullopt};
        };

        struct Ex {
            const ModelVariantPair simplex;
            const std::vector<ModsVariantPair> mods{};
        };

        const Chemistry DNA = Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ;
        const Chemistry RNA = Chemistry::RNA004_130BPS;
        auto [input, expected] = GENERATE_COPY(table<In, Ex>({
                // No version
                std::make_tuple(In{"auto"}, Ex{{MV::AUTO}}),
                std::make_tuple(In{"fast", DNA}, Ex{{MV::FAST}, {}}),
                std::make_tuple(In{"hac", DNA}, Ex{{MV::HAC}, {}}),
                std::make_tuple(In{"sup", DNA}, Ex{{MV::SUP}, {}}),

                // specific version
                std::make_tuple(In{"auto@v4.2.0", DNA}, Ex{{MV::AUTO, VV::v4_2_0}, {}}),
                std::make_tuple(In{"fast@v5.0.0", DNA}, Ex{{MV::FAST, VV::v5_0_0}, {}}),
                std::make_tuple(In{"hac@v4.2.0", DNA}, Ex{{MV::HAC, VV::v4_2_0}, {}}),
                std::make_tuple(In{"sup@v5.2.0", DNA}, Ex{{MV::SUP, VV::v5_2_0}, {}}),

                // latest version
                std::make_tuple(In{"auto@latest", RNA}, Ex{{MV::AUTO}, {}}),
                std::make_tuple(In{"fast@latest", RNA}, Ex{{MV::FAST}, {}}),
                std::make_tuple(In{"hac@latest", RNA}, Ex{{MV::HAC}, {}}),
                std::make_tuple(In{"sup@latest", RNA}, Ex{{MV::SUP}, {}}),

                // with single mods
                std::make_tuple(In{"auto,5mC"}, Ex{{MV::AUTO}, {{ModV::M_5mC}}}),
                std::make_tuple(In{"hac,4mC_5mC"}, Ex{{MV::HAC}, {{ModV::M_4mC_5mC}}}),
                std::make_tuple(In{"fast,5mC_5hmC"}, Ex{{MV::FAST}, {{ModV::M_5mC_5hmC}}}),
                std::make_tuple(In{"auto,5mCG"}, Ex{{MV::AUTO}, {{ModV::M_5mCG}}}),
                std::make_tuple(In{"hac,5mCG_5hmCG", DNA}, Ex{{MV::HAC}, {{ModV::M_5mCG_5hmCG}}}),

                std::make_tuple(In{"auto@v5.0.0,6mA"}, Ex{{MV::AUTO, VV::v5_0_0}, {{ModV::M_6mA}}}),
                std::make_tuple(In{"auto,m6A_DRACH", RNA}, Ex{{MV::AUTO}, {{ModV::M_m6A_DRACH}}}),
                std::make_tuple(In{"auto,inosine_m6A", RNA},
                                Ex{{MV::AUTO}, {{ModV::M_inosine_m6A}}}),
                std::make_tuple(In{"sup,pseU_2OmeU", RNA}, Ex{{MV::SUP}, {{ModV::M_pseU_2OmeU}}}),
                std::make_tuple(In{"sup,pseU_2OmeU,m5C_2OmeC", RNA},
                                Ex{{MV::SUP}, {{ModV::M_pseU_2OmeU}, {ModV::M_m5C_2OmeC}}}),
                // with single mods and version
                std::make_tuple(In{"sup@v4.1.0,5mC@v2"},
                                Ex{{MV::SUP, VV::v4_1_0}, {{ModV::M_5mC, VV::v2_0_0}}}),
                std::make_tuple(In{"fast@latest,5mC_5hmC@v4.0.0"},
                                Ex{{MV::FAST}, {{ModV::M_5mC_5hmC, VV::v4_0_0}}}),

                // Multi-mods
                std::make_tuple(In{"auto,5mC,6mA"}, Ex{{MV::AUTO}, {{ModV::M_5mC}, {ModV::M_6mA}}}),
                std::make_tuple(
                        In{"fast@latest,m6A_DRACH@v1,5mC_5hmC@v4.0.0"},
                        Ex{{MV::FAST},
                           {{ModV::M_m6A_DRACH, VV::v1_0_0}, {ModV::M_5mC_5hmC, VV::v4_0_0}}}),
        }));

        CATCH_CAPTURE(input.complex);
        const ModelComplex mc = ModelComplex::parse(input.complex);
        CATCH_REQUIRE(mc.is_variant_style());
        CATCH_CHECK(mc.get_simplex_model_variant() == expected.simplex);
        if (!expected.mods.empty()) {
            CATCH_CHECK(mc.get_mod_model_variants() == expected.mods);
        }
        if (input.chemistry.has_value()) {
            ModelComplexSearch search(mc, input.chemistry.value(), false);
            CATCH_CHECK(search.chemistry() == input.chemistry);
            CATCH_CHECK(search.simplex().model_type == ModelType::SIMPLEX);

            if (expected.simplex.variant != MV::AUTO) {
                CATCH_CHECK(to_string(search.simplex().simplex.variant) ==
                            to_string(expected.simplex.variant));
            } else {
                CATCH_CHECK(search.simplex().simplex.is_auto);
            }

            CATCH_CHECK(search.mods().size() == expected.mods.size());
            for (const auto &mod : search.mods()) {
                if (expected.simplex.variant != MV::AUTO) {
                    CATCH_CHECK(to_string(mod.simplex.variant) ==
                                to_string(expected.simplex.variant));
                }

                int count_mod_matches = 0;
                for (const auto &ex_mod : expected.mods) {
                    if (ex_mod.variant == mod.mods.variant) {
                        ++count_mod_matches;
                    }
                }
                CATCH_CHECK(count_mod_matches == 1);
            }
        }
    }
}

CATCH_TEST_CASE(TEST_TAG " Test BasecallerModelResolver", TEST_TAG) {
    class TestResolver final : public BasecallerModelResolver {
    public:
        TestResolver(Chemistry chem,
                     const std::string &complex,
                     const std::string &modbase_models,
                     const std::vector<std::string> &modbases)
                : BasecallerModelResolver(complex,
                                          modbase_models,
                                          modbases,
                                          std::nullopt,
                                          true,
                                          {}) {
            m_check_paths_override = false;
            m_chemistry_override = chem;
            m_download_override = [](const ModelInfo &info, [[maybe_unused]] std::string_view d) {
                return std::filesystem::path(info.name);
            };
        };
    };

    CATCH_SECTION(" test basecaller complex") {
        auto [chemistry, mvp, expected_simplex] = GENERATE(table<Chemistry, MVP, std::string>({
                std::make_tuple(Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::HAC, VV::v4_2_0},
                                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"),
                std::make_tuple(Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ, MVP{MV::SUP, VV::v5_2_0},
                                "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"),
        }));

        auto [mod] = GENERATE(table<std::optional<ModVP>>({
                std::nullopt,
                ModVP{ModsVariant::M_5mCG_5hmCG, VV::NONE},
        }));

        const auto simplex_str = to_string(mvp.variant) + "@" + to_string(mvp.ver);
        const auto complex_str = !mod.has_value() ? simplex_str
                                                  : simplex_str + "," + to_string(mod->variant) +
                                                            "@" + to_string(mod->ver);

        CATCH_CAPTURE(complex_str, to_string(chemistry));
        TestResolver complex_resolver(chemistry, complex_str, "", {});
        const ModelSources sources = complex_resolver.resolve();

        CATCH_CHECK(sources.simplex.path.filename().string() == expected_simplex);
        CATCH_CHECK(sources.simplex.info.has_value());
        CATCH_CHECK(sources.simplex.info->name == expected_simplex);
    }

    CATCH_SECTION(" test basecaller complex cli examples") {
        const auto chemistry = Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ;

        const auto src = [](const std::string &name) {
            return ModelSource{"", get_model_info(name), false};
        };

        const auto srcs = [&src](const std::string &sx, const std::vector<std::string> &mods) {
            std::vector<ModelSource> ms;
            std::transform(mods.cbegin(), mods.cend(), std::back_inserter(ms), src);
            return ModelSources{src(sx), ms, std::nullopt};
        };

        // clang-format off
        auto [expected, complex, modbases, modbases_models] = GENERATE_COPY(
            table<ModelSources, std::string, std::vector<std::string>, std::string>({
                    {srcs("dna_r10.4.1_e8.2_400bps_sup@v5.2.0", 
                        {"dna_r10.4.1_e8.2_400bps_sup@v5.2.0_5mC_5hmC@v2",}),
                        "sup@v5.2.0", {"5mC_5hmC"}, {}},
                    // Two modbase models via modbases
                    {srcs("dna_r10.4.1_e8.2_400bps_sup@v5.2.0", 
                        {"dna_r10.4.1_e8.2_400bps_sup@v5.2.0_5mC_5hmC@v2", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0_6mA@v1"}),
                        "sup@v5.2.0", {"5mC_5hmC", "6mA"}, ""},
                    {srcs("dna_r10.4.1_e8.2_400bps_hac@v5.2.0", 
                        {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2",}),
                        "hac@v5.2.0", {}, "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"},
                    // Two modbase models via modbase model paths
                    {srcs("dna_r10.4.1_e8.2_400bps_hac@v5.2.0", 
                        {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2", "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_6mA@v1"}),
                        "hac@v5.2.0", {}, "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2,dna_r10.4.1_e8.2_400bps_hac@v5.2.0_6mA@v1"},
            }));
        // clang-format on

        CATCH_CAPTURE(complex, modbases, modbases_models);
        TestResolver complex_resolver(chemistry, complex, modbases_models, modbases);
        const ModelSources sources = complex_resolver.resolve();

        CATCH_CHECK(sources.simplex.info == expected.simplex.info);
        CATCH_CHECK(sources.mods == expected.mods);
        CATCH_CHECK(sources.stereo == expected.stereo);
    }
}

CATCH_TEST_CASE(TEST_TAG " Test modbase model resolution", TEST_TAG) {
    const auto reads_path = fs::path(get_data_dir("pod5")) / "dna_r10.4.1_e8.2_400bps_5khz";
    const auto reads = dorado::utils::fetch_directory_entries(reads_path, false);

    CATCH_SECTION(" test no duplicate modbase models") {
        BasecallerModelResolver resolver{"hac",
                                         "",
                                         {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2",
                                          "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"},
                                         std::nullopt,
                                         false,
                                         reads};

        CATCH_CHECK_THROWS_AS(resolver.resolve(), std::runtime_error);
    }

    CATCH_SECTION(" test no duplicate modbase model variants") {
        BasecallerModelResolver resolver{
                "hac,5mC_5hmC,5mC_5hmC", "", {}, std::nullopt, false, reads};

        CATCH_CHECK_THROWS_AS(resolver.resolve(), std::runtime_error);
    }

    CATCH_SECTION(" test no duplicate modbase model mixed") {
        BasecallerModelResolver resolver{
                "hac,5mC_5hmC@v2", "",    {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0_5mC_5hmC@v2"},
                std::nullopt,      false, reads};

        CATCH_CHECK_THROWS_AS(resolver.resolve(), std::logic_error);
    }
}

CATCH_TEST_CASE(TEST_TAG " Test DuplexModelResolver", TEST_TAG) {
    class TestResolver final : public DuplexModelResolver {
    public:
        TestResolver(Chemistry chem, const std::string &complex)
                : DuplexModelResolver(complex, "", {}, std::nullopt, std::nullopt, true, {}) {
            m_check_paths_override = false;
            m_chemistry_override = chem;
            m_download_override = [](const ModelInfo &info, [[maybe_unused]] std::string_view d) {
                return std::filesystem::path(info.name);
            };
        };
    };

    auto [chemistry, mvp, expected_simplex, expected_stereo] =
            GENERATE(table<Chemistry, MVP, std::string, std::string>({
                    std::make_tuple(Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                                    MVP{MV::HAC, VV::v4_2_0}, "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                                    "dna_r10.4.1_e8.2_5khz_stereo@v1.1"),
                    std::make_tuple(Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                                    MVP{MV::HAC, VV::v5_0_0}, "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                                    "dna_r10.4.1_e8.2_5khz_stereo@v1.3"),
                    std::make_tuple(Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                                    MVP{MV::SUP, VV::v5_0_0}, "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
                                    "dna_r10.4.1_e8.2_5khz_stereo@v1.3"),
            }));

    auto [mod] = GENERATE(table<std::optional<ModVP>>({
            std::nullopt,
            ModVP{ModsVariant::M_5mCG_5hmCG, VV::NONE},
    }));

    const auto simplex_str = to_string(mvp.variant) + "@" + to_string(mvp.ver);
    const auto complex_str = !mod.has_value() ? simplex_str
                                              : simplex_str + "," + to_string(mod->variant) + "@" +
                                                        to_string(mod->ver);

    CATCH_CAPTURE(complex_str, to_string(chemistry));
    TestResolver complex_resolver(chemistry, complex_str);
    const ModelSources sources = complex_resolver.resolve();

    CATCH_CHECK(sources.simplex.path.filename().string() == expected_simplex);
    CATCH_CHECK(sources.simplex.info.has_value());
    CATCH_CHECK(sources.simplex.info->name == expected_simplex);

    CATCH_CHECK(sources.stereo.has_value());
    CATCH_CHECK(sources.stereo.value().info.has_value());
    CATCH_CHECK(sources.stereo->info->name == expected_stereo);
}

// }
