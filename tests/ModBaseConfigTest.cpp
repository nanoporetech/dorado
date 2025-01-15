#include "TestUtils.h"
#include "modbase/ModBaseModelConfig.h"

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#define TEST_GROUP "[modbase_config]"

namespace fs = std::filesystem;

using namespace dorado::modbase;

const auto _5mCG = get_data_dir("model_configs/dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1");
const auto _5mCG_5hmCG =
        get_data_dir("model_configs/dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2");
const auto _pseU = get_data_dir("model_configs/rna004_130bps_sup@v5.0.0_pseU@v1");
const auto _pseU_j = get_data_dir("model_configs/rna004_130bps_sup@v5.0.1_pseU@v1");
const auto _5mCG_5hmCG_v3 =
        get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v3");
const auto _6mA_v3 = get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v3");
TEST_CASE(TEST_GROUP ": modbase model parser", TEST_GROUP) {
    using Gen = ModelGeneralParams;
    using Mod = ModificationParams;
    using Ctx = ContextParams;
    using Rmt = RefinementParams;

    // For _5mCG_5hmCG kmer center index 6 to pass tests need: Log(X)/Log(4) > 7
    // -> Log(X) > 7*1.387 -> X := Exp(9.71) = 16482
    std::vector<float> refined_kmer_levels(16482, 0);
    auto [path, general, mods, context, refine] =
            // clang-format off
            GENERATE_COPY(table<fs::path, Gen, Mod, Ctx, Rmt>({
            std::make_tuple(
                _5mCG, 
                Gen{ModelType::CONV_V1, 64, 9, 2, 3}, 
                Mod{{"m"}, {"5mC"}, "CG", 0},
                Ctx{50, 50, 100, 4, 4, false, false}, 
                Rmt{}
            ),
            std::make_tuple(
                _5mCG_5hmCG, 
                Gen{ModelType::CONV_LSTM_V1, 256, 9, 3, 3},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0,},                
                Ctx{50, 50, 100, 4, 4, false, false}, 
                Rmt{7, 6, refined_kmer_levels}
            ),
            std::make_tuple(
                _pseU,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, 3},
                Mod{{"17802"}, {"pseU"}, "T", 0},            
                Ctx{150, 150, 300, 4, 4, true, false},
                Rmt{7, 3, refined_kmer_levels}
            ),
            std::make_tuple(
                _pseU_j,
                Gen{ModelType::CONV_LSTM_V1, 128, 9, 2, 3},
                Mod{{"17802"}, {"pseU"}, "T", 0},            
                Ctx{150, 150, 300, 4, 4, true, true},
                Rmt{7, 3, refined_kmer_levels}
            ),
            std::make_tuple(
                _5mCG_5hmCG_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 3, 6},
                Mod{{"h", "m"}, {"5hmC", "5mC"}, "CG", 0},            
                Ctx{96, 96, 192, 4, 4, false, true},
                Rmt{7, 6, refined_kmer_levels}
            ),
            std::make_tuple(
                _6mA_v3,
                Gen{ModelType::CONV_LSTM_V2, 256, 9, 2, 3},
                Mod{{"a"}, {"6mA"}, "A", 0},            
                Ctx{150, 150, 600, 4, 4, false, true},
                Rmt{7, 6, refined_kmer_levels}
            ),
            }));
    // clang-format on
    const auto model = test::load_modbase_model_config(path, refined_kmer_levels);

    SECTION("general model parameters") {
        const auto& g = model.general;
        CAPTURE(path, to_string(g.model_type), g.size, g.kmer_len, g.num_out);
        CHECK(g.model_type == general.model_type);
        CHECK(g.size == general.size);
        CHECK(g.kmer_len == general.kmer_len);
        CHECK(g.num_out == general.num_out);
    }

    SECTION("modification parameters") {
        const auto& m = model.mods;
        CAPTURE(path, m.codes, m.long_names, m.count, m.motif, m.motif_offset);
        CHECK(m.codes == mods.codes);
        CHECK(m.long_names == mods.long_names);
        CHECK(m.count == mods.count);
        CHECK(m.motif == mods.motif);
        CHECK(m.motif_offset == mods.motif_offset);
    }

    SECTION("context parameters") {
        const auto& c = model.context;
        CAPTURE(path, c.samples_before, c.samples_after, c.samples, c.bases_before, c.bases_after,
                c.kmer_len, c.reverse, c.base_start_justify);
        CHECK(c.samples_before == context.samples_before);
        CHECK(c.samples_after == context.samples_after);
        CHECK(c.samples == context.samples);
        CHECK(c.bases_before == context.bases_before);
        CHECK(c.bases_after == context.bases_after);
        CHECK(c.kmer_len == context.kmer_len);
        CHECK(c.reverse == context.reverse);
        CHECK(c.base_start_justify == context.base_start_justify);
        CHECK(c.chunk_size == context.chunk_size);
    }

    SECTION("refinement parameters") {
        const auto& r = model.refine;
        CAPTURE(path, r.do_rough_rescale, r.kmer_len, r.center_idx);
        CHECK(r.do_rough_rescale == refine.do_rough_rescale);
        CHECK(r.kmer_len == refine.kmer_len);
        CHECK(r.center_idx == refine.center_idx);
        CHECK(r.levels.size() == refine.levels.size());
    }
}
