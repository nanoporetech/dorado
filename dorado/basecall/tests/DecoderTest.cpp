#include "decode/Decoder.h"

#include "TestUtils.h"
#include "config/BasecallModelConfig.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#define CUT_TAG "[Decoder]"

namespace {

dorado::config::BasecallModelConfig load_config(const char* model_name) {
    return dorado::config::load_model_config(std::filesystem::path(get_data_dir("model_configs")) /
                                             model_name);
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " decoder options use model q-score calibration", CUT_TAG) {
    const auto config = load_config("dna_r10.4.1_e8.2_400bps_hac@v4.2.0");

    const auto options = dorado::basecall::decode::create_decoder_options(config);

    CATCH_CHECK(options.q_shift == config.qbias);
    CATCH_CHECK(options.q_scale == config.qscale);
    CATCH_CHECK(options.beam_width == dorado::basecall::decode::DecoderOptions{}.beam_width);
    CATCH_CHECK(options.beam_cut == dorado::basecall::decode::DecoderOptions{}.beam_cut);
    CATCH_CHECK(options.blank_score == dorado::basecall::decode::DecoderOptions{}.blank_score);
}
