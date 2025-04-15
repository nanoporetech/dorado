#include "TestUtils.h"
#include "config/ModBaseModelConfig.h"
#include "config/common.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>

#define TEST_GROUP "[modbase_config]"

namespace fs = std::filesystem;

using namespace dorado::config;

namespace {
void check_conv_params(const ConvParams& c, const ConvParams& expect) {
    CATCH_CAPTURE(c.to_string());
    CATCH_CHECK(c.insize == expect.insize);
    CATCH_CHECK(c.size == expect.size);
    CATCH_CHECK(c.activation == expect.activation);
    CATCH_CHECK(c.stride == expect.stride);
    CATCH_CHECK(c.winlen == expect.winlen);
}
}  // namespace

const auto _5mCG = get_data_dir("model_configs/dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1");
const auto _5mCG_5hmCG =
        get_data_dir("model_configs/dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2");
const auto _pseU = get_data_dir("model_configs/rna004_130bps_sup@v5.0.0_pseU@v1");
const auto _pseU_j = get_data_dir("model_configs/rna004_130bps_sup@v5.0.1_pseU@v1");
const auto _5mCG_5hmCG_v3 =
        get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v3");
const auto _6mA_v3 = get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v3");
const auto _6mA_v4 = get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v4");

const ModulesParams modules_6mA_v4{
        // clang-format off
    {   // Sequence Convs
        ConvParams{36, 16, 5, 1, Activation::TANH}, 
        ConvParams{16, 128, 16, 1, Activation::TANH}},
    {   // Signal Convs
        ConvParams{1, 4, 5, 1, Activation::TANH}, 
        ConvParams{4, 16, 5, 1, Activation::TANH},
        ConvParams{16, 128, 16, 6, Activation::TANH}},
    // Merge
    ConvParams{256, 384, 5, 1, Activation::TANH}, 
    {   // LSTMS
        LSTMParams{384, 0}, 
        LSTMParams{384, 1}
    },
    LinearParams{384, 2},  
    LinearUpsampleParams{2, 2}};
// clang-format on

CATCH_TEST_CASE(TEST_GROUP ": modbase model parser", TEST_GROUP) {
    using Gen = ModelGeneralParams;
    using Mod = ModificationParams;
    using Ctx = ContextParams;
    using Rmt = RefinementParams;

    auto [path, general, mods, context, refine] =
            // clang-format off
            GENERATE_COPY(table<fs::path, Gen, Mod, Ctx, Rmt>({
            std::make_tuple(
                _5mCG, 
                Gen{ModelType::CONV_V1, 64, 9, 2, 3, std::nullopt}, 
                Mod{{"m"}, {"5mC"}, "CG", 0},
                Ctx{50, 50, 100, 4, 4, false, false}, 
                Rmt{}
            ),
            std::make_tuple(
                _5mCG_5hmCG, 
                Gen{ModelType::CONV_LSTM_V1, 256, 9, 3, 3, std::nullopt},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0,},                
                Ctx{50, 50, 100, 4, 4, false, false}, 
                Rmt{6}
            ),
            std::make_tuple(
                _pseU,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, 3, std::nullopt},
                Mod{{"17802"}, {"pseU"}, "T", 0},            
                Ctx{150, 150, 300, 4, 4, true, false},
                Rmt{3}
            ),
            std::make_tuple(
                _pseU_j,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, 3, std::nullopt},
                Mod{{"17802"}, {"pseU"}, "T", 0},            
                Ctx{150, 150, 300, 4, 4, true, true},
                Rmt{3}
            ),
            std::make_tuple(
                _5mCG_5hmCG_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 3, 6, std::nullopt},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0},            
                Ctx{96, 96, 192, 4, 4, false, true},
                Rmt{6}
            ),
            std::make_tuple(
                _6mA_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 2, 3, std::nullopt},
                Mod{{"a"}, {"6mA"}, "A", 0},            
                Ctx{150, 150, 600, 4, 4, false, true},
                Rmt{6}
            ),
            std::make_tuple(
                _6mA_v4,
                Gen{ModelType::CONV_LSTM_V3, 384, 9, 2, 6, modules_6mA_v4},
                Mod{{"a"}, {"6mA"}, "A", 0},            
                Ctx{150, 150, 600, 4, 4, false, true},
                Rmt{6}
            ),
            }));
    // clang-format on
    const auto model = load_modbase_model_config(path);

    CATCH_SECTION("general model parameters") {
        const auto& g = model.general;
        CATCH_CAPTURE(path, to_string(g.model_type), g.size, g.kmer_len, g.num_out);
        CATCH_CHECK(g.model_type == general.model_type);
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
        const auto& m = model.mods;
        CATCH_CAPTURE(path, m.codes, m.long_names, m.count, m.motif, m.motif_offset);
        CATCH_CHECK(m.codes == mods.codes);
        CATCH_CHECK(m.long_names == mods.long_names);
        CATCH_CHECK(m.count == mods.count);
        CATCH_CHECK(m.motif == mods.motif);
        CATCH_CHECK(m.motif_offset == mods.motif_offset);
    }

    CATCH_SECTION("context parameters") {
        const auto& c = model.context;
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
        CATCH_CHECK(c.chunk_size == context.chunk_size);
    }

    CATCH_SECTION("refinement parameters") {
        const auto& r = model.refine;
        CATCH_CAPTURE(path, r.do_rough_rescale, r.center_idx);
        CATCH_CHECK(r.do_rough_rescale == refine.do_rough_rescale);
        CATCH_CHECK(r.center_idx == refine.center_idx);
        CATCH_CHECK(r.center_idx < static_cast<size_t>(model.general.kmer_len));
    }
}
