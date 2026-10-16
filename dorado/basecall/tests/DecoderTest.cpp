#include "decode/Decoder.h"

#include "TestUtils.h"
#include "config/BasecallModelConfig.h"
#include "decode/CPUDecoder.h"

#include <ATen/ops/zeros.h>
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

CATCH_TEST_CASE(CUT_TAG " decoder config describes the model state space", CUT_TAG) {
    CATCH_SECTION("LSTM alphabet") {
        auto config = load_config("dna_r10.4.1_e8.2_400bps_hac@v4.2.0");
        config.state_len = 3;
        config.clamp = true;

        const auto decoder_config = dorado::basecall::decode::create_decoder_config(config);

        CATCH_CHECK(decoder_config.num_bases == 4);
        CATCH_CHECK(decoder_config.state_len == 3);
        CATCH_CHECK(decoder_config.num_states() == 64);
        CATCH_CHECK(decoder_config.num_transitions() == 256);
        CATCH_CHECK(decoder_config.score_clamp == 5.f);
    }

    CATCH_SECTION("transformer alphabet") {
        auto config = load_config("dna_r10.4.1_e8.2_400bps_sup@v5.0.0");
        config.tx->crf.n_base = 5;
        config.state_len = 2;
        config.clamp = false;

        const auto decoder_config = dorado::basecall::decode::create_decoder_config(config);

        CATCH_CHECK(decoder_config.num_bases == 5);
        CATCH_CHECK(decoder_config.num_states() == 25);
        CATCH_CHECK(decoder_config.num_transitions() == 125);
        CATCH_CHECK(decoder_config.score_clamp == 0.f);
    }
}

CATCH_TEST_CASE(CUT_TAG " CPU decoder preserves the typed phase boundary", CUT_TAG) {
    constexpr int num_timesteps = 4;
    constexpr int num_chunks = 1;
    constexpr int num_transitions = 16;
    auto scores = at::zeros({num_timesteps, num_chunks, num_transitions});
    const dorado::basecall::decode::DecoderOptions options{};
    const dorado::basecall::decode::CPUDecoder decoder;

    auto prepared = decoder.prepare_decode({scores, num_chunks, options});

    CATCH_CHECK(prepared.data.is_same(scores));
    CATCH_CHECK(prepared.num_chunks == num_chunks);
    CATCH_CHECK(prepared.aux == nullptr);

    const auto chunks = decoder.decode(prepared);
    CATCH_REQUIRE(chunks.size() == num_chunks);
    CATCH_CHECK(chunks.front().moves.size() == num_timesteps);
    CATCH_CHECK(chunks.front().sequence.size() == chunks.front().qstring.size());
}
