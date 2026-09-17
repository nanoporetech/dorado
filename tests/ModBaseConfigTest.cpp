#include "TestUtils.h"
#include "config/ModBaseModelConfig.h"
#include "config/common.h"
#include "utils/overloaded.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>

#define TEST_GROUP "[modbase_config]"

using namespace dorado::config;
using namespace dorado::config::modbase;

namespace {

struct General {
    ModelType model_type;
    int size;
    int kmer_len;
    int num_out;
    std::optional<ModulesParams> modules;
};

General get_general(const ModBaseModelConfig& config) {
    return std::visit(
            dorado::utils::overloaded{
                    [](const ConvOnlyModelConfig& model) -> General {
                        return {
                                ModelType::CONV_V1,    model.general.size, model.general.kmer_len,
                                model.general.num_out, std::nullopt,
                        };
                    },
                    [](const ConvLSTMModelConfig& model) -> General {
                        return {
                                ModelType::CONV_LSTM_V1, model.general.size, model.general.kmer_len,
                                model.general.num_out,   std::nullopt,
                        };
                    },
                    [](const ConvLSTMV2ModelConfig& model) -> General {
                        return {
                                ModelType::CONV_LSTM_V2, model.general.size, model.general.kmer_len,
                                model.general.num_out,   std::nullopt,
                        };
                    },
                    [](const ConvLSTMV3ModelConfig& model) -> General {
                        return {
                                ModelType::CONV_LSTM_V3, model.general.size, model.general.kmer_len,
                                model.general.num_out,   model.modules,
                        };
                    },
                    [](const LatentSpaceV1ModelConfig& model) -> General {
                        return {
                                ModelType::LATENT_SPACE_V1,
                                model.general.size,
                                model.general.kmer_len,
                                model.general.num_out,
                                std::nullopt,
                        };
                    },
                    [](const auto&) -> General { throw std::logic_error("Invalid config type."); },
            },
            config);
}

void check_conv_params(const ConvParams& c, const ConvParams& expect) {
    CATCH_CAPTURE(c.to_string());
    CATCH_CHECK(c.insize == expect.insize);
    CATCH_CHECK(c.size == expect.size);
    CATCH_CHECK(c.activation == expect.activation);
    CATCH_CHECK(c.stride == expect.stride);
    CATCH_CHECK(c.winlen == expect.winlen);
}
}  // namespace

const char config_5mCG_5hmCG[] = "dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2";
const char config_pseU[] = "rna004_130bps_sup@v5.0.0_pseU@v1";
const char config_pseU_j[] = "rna004_130bps_sup@v5.0.1_pseU@v1";
const char config_5mCG_5hmCG_v3[] = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v3";
const char config_6mA_v3[] = "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v3";
const char config_6mA_v4[] = "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v4";
const char config_5mCG_conv_only[] = "dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1";
const char config_6mA_latent_space[] = "latent_space_6mA";
const char config_6mA_latent_space_multilayer[] = "latent_space_6mA_multilayer";

const ModulesParams modules_6mA_v4{
        // Sequence Convs
        {
                ConvParams{36, 16, 5, 1, Activation::TANH, 0},
                ConvParams{16, 128, 16, 1, Activation::TANH, 0},
        },
        // Signal Convs
        {
                ConvParams{1, 4, 5, 1, Activation::TANH, 0},
                ConvParams{4, 16, 5, 1, Activation::TANH, 0},
                ConvParams{16, 128, 16, 6, Activation::TANH, 0},
        },
        // Merge
        ConvParams{256, 384, 5, 1, Activation::TANH, 0},
        // LSTMS
        {
                LSTMParams{384, 0},
                LSTMParams{384, 1},
        },
        LinearParams{384, 2},
        LinearUpsampleParams{2, 2},
};

CATCH_TEST_CASE(TEST_GROUP ": modbase model parser", TEST_GROUP) {
    using Gen = General;
    using Mod = ModificationParams;
    using Ctx = ContextParams;
    using Rmt = RefinementParams;

    auto [config, general, mods, context, refine] =
            // clang-format off
            GENERATE_COPY(table<std::string, Gen, Mod, Ctx, Rmt>({
            std::make_tuple(
                config_5mCG_conv_only,
                Gen{ModelType::CONV_V1, 64, 9, 2, std::nullopt},
                Mod{{"m"}, {"5mC"}, "CG", 0},
                Ctx{50, 50, 4, 4, false, false},
                Rmt{}
            ),
            std::make_tuple(
                config_5mCG_5hmCG,
                Gen{ModelType::CONV_LSTM_V1, 256, 9, 3, std::nullopt},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0},
                Ctx{50, 50, 4, 4, false, false},
                Rmt{6}
            ),
            std::make_tuple(
                config_pseU,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, std::nullopt},
                Mod{{"17802"}, {"pseU"}, "T", 0},
                Ctx{150, 150, 4, 4, true, false},
                Rmt{3}
            ),
            std::make_tuple(
                config_pseU_j,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, std::nullopt},
                Mod{{"17802"}, {"pseU"}, "T", 0},
                Ctx{150, 150, 4, 4, true, true},
                Rmt{3}
            ),
            std::make_tuple(
                config_5mCG_5hmCG_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 3, std::nullopt},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0},
                Ctx{96, 96, 4, 4, false, true},
                Rmt{6}
            ),
            std::make_tuple(
                config_6mA_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 2, std::nullopt},
                Mod{{"a"}, {"6mA"}, "A", 0},
                Ctx{150, 150, 4, 4, false, true},
                Rmt{6}
            ),
            std::make_tuple(
                config_6mA_v4,
                Gen{ModelType::CONV_LSTM_V3, 384, 9, 2, modules_6mA_v4},
                Mod{{"a"}, {"6mA"}, "A", 0},
                Ctx{150, 150, 4, 4, false, true},
                Rmt{6}
            ),
            std::make_tuple(
                config_6mA_latent_space,
                Gen{ModelType::LATENT_SPACE_V1, 64, 9, 2, std::nullopt},
                Mod{{"a"}, {"6mA"}, "A", 0},
                Ctx{150, 150, 4, 4, false, true},
                Rmt{}
            ),
            }));
    // clang-format on
    const auto path = get_data_dir("model_configs") / config;
    const auto model = load_modbase_model_config(path);

    CATCH_SECTION("general model parameters") {
        const auto g = get_general(model);
        CATCH_CAPTURE(path, to_string(g.model_type), g.size, g.kmer_len, g.num_out);
        CATCH_CHECK(g.model_type == general.model_type);
        CATCH_CHECK(get_model_type(model) == general.model_type);
        CATCH_CHECK(g.size == general.size);
        CATCH_CHECK(g.kmer_len == general.kmer_len);
        CATCH_CHECK(g.num_out == general.num_out);

        if (g.modules.has_value()) {
            CATCH_SECTION("sequence convs") {
                CATCH_REQUIRE(g.modules->sequence_convs.size() ==
                              general.modules->sequence_convs.size());
                for (size_t i = 0; i < g.modules->sequence_convs.size(); i++) {
                    check_conv_params(g.modules->sequence_convs.at(i),
                                      general.modules->sequence_convs.at(i));
                }
            }
            CATCH_SECTION("signal convs") {
                CATCH_REQUIRE(g.modules->signal_convs.size() ==
                              general.modules->signal_convs.size());
                for (size_t i = 0; i < g.modules->signal_convs.size(); i++) {
                    check_conv_params(g.modules->signal_convs.at(i),
                                      general.modules->signal_convs.at(i));
                }
            }
            CATCH_SECTION("merge conv") {
                check_conv_params(g.modules->merge_conv, general.modules->merge_conv);
            }
            CATCH_SECTION("lstms") {
                CATCH_REQUIRE(g.modules->lstms.size() == general.modules->lstms.size());
                for (size_t i = 0; i < g.modules->lstms.size(); i++) {
                    const auto& l = g.modules->lstms.at(i);
                    const auto& e = general.modules->lstms.at(i);
                    CATCH_CHECK(l.reverse == e.reverse);
                    CATCH_CHECK(l.size == e.size);
                }
            }
        }
    }

    CATCH_SECTION("modification parameters") {
        const auto& m = get_model_config_data(model).mods;
        CATCH_CAPTURE(path, m.codes, m.long_names, m.count, m.motif, m.motif_offset);
        CATCH_CHECK(m.codes == mods.codes);
        CATCH_CHECK(m.long_names == mods.long_names);
        CATCH_CHECK(m.count == mods.count);
        CATCH_CHECK(m.motif == mods.motif);
        CATCH_CHECK(m.motif_offset == mods.motif_offset);
    }

    CATCH_SECTION("context parameters") {
        const auto& c = get_model_config_data(model).context;
        CATCH_CAPTURE(path, c.samples_before, c.samples_after, c.samples, c.bases_before,
                      c.bases_after, c.kmer_len, c.reverse, c.base_start_justify);
        CATCH_CHECK(c.samples_before == context.samples_before);
        CATCH_CHECK(c.samples_after == context.samples_after);
        CATCH_CHECK(c.samples == context.samples);
        CATCH_CHECK(c.bases_before == context.bases_before);
        CATCH_CHECK(c.bases_after == context.bases_after);
        CATCH_CHECK(c.kmer_len == context.kmer_len);
        CATCH_CHECK(c.reverse == context.reverse);
        CATCH_CHECK(c.base_start_justify == context.base_start_justify);
    }

    CATCH_SECTION("refinement parameters") {
        const auto& r = get_model_config_data(model).refine;
        CATCH_CAPTURE(path, r.do_rough_rescale, r.center_idx);
        CATCH_CHECK(r.do_rough_rescale == refine.do_rough_rescale);
        CATCH_CHECK(r.center_idx == refine.center_idx);
        CATCH_CHECK(r.center_idx < static_cast<size_t>(general.kmer_len));
    }
}

CATCH_TEST_CASE(TEST_GROUP ": latent space model parser", TEST_GROUP) {
    const auto path = get_data_dir("model_configs") / config_6mA_latent_space;
    const auto model = load_modbase_model_config(path);
    const auto& latent = std::get<LatentSpaceV1ModelConfig>(model);

    CATCH_CHECK(get_modbase_model_type(path) == ModelType::LATENT_SPACE_V1);
    CATCH_CHECK(to_string(ModelType::LATENT_SPACE_V1) == "latent_space_v1");
    CATCH_CHECK(is_chunked_input_model(model));
    CATCH_CHECK(latent.chunked.chunk_size == 1536);
    CATCH_CHECK(latent.sequence_stride == 1);

    const auto& modules = latent.modules;
    CATCH_REQUIRE(modules.sequence_convs.size() == 1);
    check_conv_params(modules.sequence_convs.front(),
                      ConvParams{36, 64, 5, 1, Activation::TANH, 0});
    CATCH_REQUIRE(modules.projection.size() == 1);
    CATCH_CHECK(modules.projection.front().in_size == 1024);
    CATCH_CHECK(modules.projection.front().out_size == 64);
    CATCH_REQUIRE(modules.classifier.size() == 1);
    CATCH_CHECK(modules.classifier.front().in_size == 64);
    CATCH_CHECK(modules.classifier.front().out_size == 2);

    CATCH_REQUIRE(modules.transformer_encoder.size() == 3);
    for (const auto& layer : modules.transformer_encoder) {
        CATCH_CHECK(layer.d_model == 64);
        CATCH_CHECK(layer.nhead == 4);
        CATCH_CHECK(layer.dim_feedforward == 128);
        CATCH_CHECK(layer.deepnorm_alpha == 1.5650845f);
        CATCH_CHECK(layer.attn_window == std::pair<int, int>{15, 16});
    }
}

CATCH_TEST_CASE(TEST_GROUP ": latent space model parser supports multilayer modules", TEST_GROUP) {
    const auto path = get_data_dir("model_configs") / config_6mA_latent_space_multilayer;
    const auto model = load_modbase_model_config(path);
    const auto& latent = std::get<LatentSpaceV1ModelConfig>(model);
    const auto& modules = latent.modules;

    CATCH_REQUIRE(modules.sequence_convs.size() == 2);
    check_conv_params(modules.sequence_convs.at(0), ConvParams{36, 32, 5, 1, Activation::TANH, 0});
    check_conv_params(modules.sequence_convs.at(1), ConvParams{32, 64, 3, 1, Activation::TANH, 0});

    CATCH_REQUIRE(modules.projection.size() == 2);
    CATCH_CHECK(modules.projection.at(0).in_size == 1024);
    CATCH_CHECK(modules.projection.at(0).out_size == 32);
    CATCH_CHECK(modules.projection.at(1).in_size == 32);
    CATCH_CHECK(modules.projection.at(1).out_size == 64);

    CATCH_REQUIRE(modules.classifier.size() == 2);
    CATCH_CHECK(modules.classifier.at(0).in_size == 64);
    CATCH_CHECK(modules.classifier.at(0).out_size == 16);
    CATCH_CHECK(modules.classifier.at(1).in_size == 16);
    CATCH_CHECK(modules.classifier.at(1).out_size == 2);
}
