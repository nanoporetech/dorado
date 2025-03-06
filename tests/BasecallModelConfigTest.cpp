#include "config/BasecallModelConfig.h"

#include "TestUtils.h"
#include "models/kits.h"
#include "utils/parameters.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <string>

#define CUT_TAG "[BasecallModelConfig]"

using namespace dorado::config;
using SampleType = dorado::models::SampleType;

namespace fs = std::filesystem;

CATCH_TEST_CASE(CUT_TAG ": test normalise BatchParams", CUT_TAG) {
    CATCH_SECTION("test on defaults") {
        const fs::path path =
                fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
        BasecallModelConfig config = load_model_config(path);
        config.normalise_basecaller_params();
        CATCH_CHECK(config.has_normalised_basecaller_params());
    }

    CATCH_SECTION("test known non-normalised") {
        const fs::path path =
                fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
        BasecallModelConfig config = load_model_config(path);

        // Set chunksize to (12 * 16 * 10) + 1 to ensure it's not mod192
        config.basecaller.set_chunk_size(1921);
        CATCH_CHECK_FALSE(config.has_normalised_basecaller_params());

        config.normalise_basecaller_params();
        CATCH_CHECK(config.has_normalised_basecaller_params());
        CATCH_CHECK(config.basecaller.chunk_size() % config.stride_inner() == 0);
        // Expected (1921 / 192) * 192
        CATCH_CHECK(config.basecaller.chunk_size() == 1920);
    }
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r10.4.1 sup@v5.0.0 transformer model load", CUT_TAG) {
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == true);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 6);
    CATCH_CHECK(config.lstm_size == -1);
    CATCH_CHECK(config.scale == 1.0f);
    CATCH_CHECK(config.state_len == 5);
    CATCH_CHECK(config.outsize == 4096);
    CATCH_CHECK(config.clamp == false);
    CATCH_CHECK(config.out_features.has_value());
    CATCH_CHECK(config.out_features.value() == 4096);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.stride_inner() == 12);
    CATCH_CHECK(config.scale_factor() == 2);
    CATCH_CHECK(config.stride_inner() == config.stride * config.scale_factor());

    CATCH_CHECK(config.qbias == 0.0f);
    CATCH_CHECK(config.qscale == 1.0f);
    CATCH_CHECK(config.sample_rate == 5000);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::PA;
    sig.standardisation.standardise = true;
    sig.standardisation.mean = 93.6376f;
    sig.standardisation.stdev = 22.6004f;

    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    CATCH_CHECK(config.convs.size() == 5);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 64);
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 64);
    CATCH_CHECK(conv2.size == 64);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH);
    CATCH_CHECK(conv3.insize == 64);
    CATCH_CHECK(conv3.size == 128);
    CATCH_CHECK(conv3.stride == 3);
    CATCH_CHECK(conv3.winlen == 9);

    ConvParams conv4 = config.convs[3];
    CATCH_CHECK(conv4.activation == Activation::SWISH);
    CATCH_CHECK(conv4.insize == 128);
    CATCH_CHECK(conv4.size == 128);
    CATCH_CHECK(conv4.stride == 2);
    CATCH_CHECK(conv4.winlen == 9);

    ConvParams conv5 = config.convs[4];
    CATCH_CHECK(conv5.activation == Activation::SWISH);
    CATCH_CHECK(conv5.insize == 128);
    CATCH_CHECK(conv5.size == 512);
    CATCH_CHECK(conv5.stride == 2);
    CATCH_CHECK(conv5.winlen == 5);

    CATCH_CHECK(config.tx.has_value());

    CATCH_CHECK(config.tx->tx.d_model == 512);
    CATCH_CHECK(config.tx->tx.depth == 18);
    CATCH_CHECK(config.tx->tx.dim_feedforward == 2048);
    CATCH_CHECK(config.tx->tx.attn_window == std::pair<int, int>{127, 128});

    CATCH_CHECK(config.tx->crf.insize == 512);
    CATCH_CHECK(config.tx->crf.n_base == 4);
    CATCH_CHECK(config.tx->crf.state_len == 5);
    CATCH_CHECK(config.tx->crf.blank_score == 2.0);
    CATCH_CHECK(config.tx->crf.scale == 5.0);

    CATCH_CHECK(config.tx->upsample.scale_factor == 2);
    CATCH_CHECK(config.tx->upsample.d_model == 512);

    CATCH_CHECK(config.basecaller.chunk_size() == 12288);
    CATCH_CHECK(config.basecaller.overlap() == 600);
    // Model config basecaller.batchsize is always ignored - expect default
    CATCH_CHECK(config.basecaller.batch_size() == dorado::utils::default_parameters.batchsize);

    // We know that chunksize and overlap over stride inner are zero (12288 % (12 * 16) && 600 % 12)
    CATCH_CHECK(config.has_normalised_basecaller_params());
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r9.4.1 hac@v3.3 model load", CUT_TAG) {
    const fs::path path = fs::path(get_data_dir("model_configs/dna_r9.4.1_e8_hac@v3.3"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == true);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 5);
    CATCH_CHECK(config.lstm_size == 384);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 5.0f);
    CATCH_CHECK(config.state_len == 4);
    CATCH_CHECK(config.outsize == 1024);
    CATCH_CHECK(config.clamp == false);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.stride_inner() == 5);
    CATCH_CHECK(config.scale_factor() == 1);
    CATCH_CHECK(config.stride_inner() == config.stride * config.scale_factor());

    CATCH_CHECK(config.qbias == -0.1721f);
    CATCH_CHECK(config.qscale == 0.9356f);
    CATCH_CHECK(config.sample_rate == -1);

    SignalNormalisationParams sig;
    // r9.4.1 model does not use quantile scaling
    sig.strategy = ScalingStrategy::MED_MAD;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 4);  // default first_conv value
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 4);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 384);
    CATCH_CHECK(conv3.stride == 5);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r10.4.1 fast@v4.0.0 model load", CUT_TAG) {
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_260bps_fast@v4.0.0"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == false);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 5);
    CATCH_CHECK(config.lstm_size == 96);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 1.0f);
    CATCH_CHECK(config.state_len == 3);
    CATCH_CHECK(config.outsize == 256);
    CATCH_CHECK(config.clamp == false);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.qbias == -3.0f);
    CATCH_CHECK(config.qscale == 1.04f);
    CATCH_CHECK(config.sample_rate == -1);

    SignalNormalisationParams sig;
    // r9.4.1 model does not use quantile scaling
    sig.strategy = ScalingStrategy::QUANTILE;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 16);  // default first_conv value
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 16);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 96);
    CATCH_CHECK(conv3.stride == 5);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r10.4.1 hac@v4.2.0 model load", CUT_TAG) {
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_hac@v4.2.0"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == true);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 6);
    CATCH_CHECK(config.lstm_size == 384);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 1.0f);
    CATCH_CHECK(config.state_len == 4);
    CATCH_CHECK(config.outsize == 1024);
    CATCH_CHECK(config.clamp == true);
    CATCH_CHECK(config.out_features.has_value());
    CATCH_CHECK(config.out_features.value() == 128);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.qbias == -0.2f);
    CATCH_CHECK(config.qscale == 0.95f);
    CATCH_CHECK(config.sample_rate == 5000);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::QUANTILE;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH_CLAMP);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 16);
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH_CLAMP);
    CATCH_CHECK(conv2.insize == 16);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH_CLAMP);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 384);
    CATCH_CHECK(conv3.stride == 6);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r10.4.1 hac@v4.3.0 pa model load", CUT_TAG) {
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_hac@v4.3.0"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == false);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 6);
    CATCH_CHECK(config.lstm_size == 384);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 1.0f);
    CATCH_CHECK(config.state_len == 4);
    CATCH_CHECK(config.outsize == 1024);
    CATCH_CHECK(config.clamp == true);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.qbias == -1.1f);
    CATCH_CHECK(config.qscale == 1.1f);
    CATCH_CHECK(config.sample_rate == 5000);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::PA;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == true);
    CATCH_CHECK(ssp.mean == 91.88f);
    CATCH_CHECK(ssp.stdev == 22.65f);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 16);
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 16);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::TANH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 384);
    CATCH_CHECK(conv3.stride == 6);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test dna_r10.4.1 hac@v4.3.0 quantile model load", CUT_TAG) {
    const fs::path path =
            fs::path(get_data_dir("model_configs/dna_r10.4.1_e8.2_400bps_hac@v4.3.0_quantile"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == false);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 6);
    CATCH_CHECK(config.lstm_size == 384);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 1.0f);
    CATCH_CHECK(config.state_len == 4);
    CATCH_CHECK(config.outsize == 1024);
    CATCH_CHECK(config.clamp == true);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::DNA);

    CATCH_CHECK(config.qbias == 0.0f);
    CATCH_CHECK(config.qscale == 1.0f);
    CATCH_CHECK(config.sample_rate == -1);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::QUANTILE;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == sig.quantile.quantile_a);
    CATCH_CHECK(qsp.quantile_b == sig.quantile.quantile_b);
    CATCH_CHECK(qsp.scale_multiplier == sig.quantile.scale_multiplier);
    CATCH_CHECK(qsp.shift_multiplier == sig.quantile.shift_multiplier);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 16);
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 16);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::TANH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 384);
    CATCH_CHECK(conv3.stride == 6);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test rna002 fast@v3 model load", CUT_TAG) {
    const fs::path path = fs::path(get_data_dir("model_configs/rna002_70bps_fast@v3"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == true);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 5);
    CATCH_CHECK(config.lstm_size == 96);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 5.0f);
    CATCH_CHECK(config.state_len == 3);
    CATCH_CHECK(config.outsize == 256);
    CATCH_CHECK(config.clamp == false);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::RNA002);

    CATCH_CHECK(config.qbias == -1.6f);
    CATCH_CHECK(config.qscale == 1.0f);
    CATCH_CHECK(config.sample_rate == 3000);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::QUANTILE;
    CATCH_CHECK(config.signal_norm_params.strategy == sig.strategy);

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == 0.2f);
    CATCH_CHECK(qsp.quantile_b == 0.8f);
    CATCH_CHECK(qsp.scale_multiplier == 0.59f);
    CATCH_CHECK(qsp.shift_multiplier == 0.48f);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 4);  // default first_conv value
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 4);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 96);
    CATCH_CHECK(conv3.stride == 5);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test rna004 sup@v3.0.1 model load", CUT_TAG) {
    const fs::path path = fs::path(get_data_dir("model_configs/rna004_130bps_sup@v3.0.1"));
    const BasecallModelConfig config = load_model_config(path);

    CATCH_CHECK(config.model_path == path);
    CATCH_CHECK(config.bias == true);
    CATCH_CHECK(config.num_features == 1);
    CATCH_CHECK(config.stride == 5);
    CATCH_CHECK(config.lstm_size == 768);
    CATCH_CHECK(config.blank_score == 2.0f);
    CATCH_CHECK(config.scale == 5.0f);
    CATCH_CHECK(config.state_len == 5);
    CATCH_CHECK(config.outsize == 4096);
    CATCH_CHECK(config.clamp == false);
    CATCH_CHECK(config.out_features.has_value() == false);
    CATCH_CHECK(config.sample_type == SampleType::RNA004);

    CATCH_CHECK(config.qbias == -0.1f);
    CATCH_CHECK(config.qscale == 0.9f);
    CATCH_CHECK(config.sample_rate == 4000);

    SignalNormalisationParams sig;
    sig.strategy = ScalingStrategy::QUANTILE;
    CATCH_CHECK(config.signal_norm_params.strategy == ScalingStrategy::QUANTILE);
    // Intentionally edited values from default values to test parser

    const QuantileScalingParams &qsp = config.signal_norm_params.quantile;
    CATCH_CHECK(qsp.quantile_a == 0.22f);
    CATCH_CHECK(qsp.quantile_b == 0.88f);
    CATCH_CHECK(qsp.scale_multiplier == 0.595f);
    CATCH_CHECK(qsp.shift_multiplier == 0.485f);

    const StandardisationScalingParams &ssp = config.signal_norm_params.standardisation;
    CATCH_CHECK(ssp.standardise == false);
    CATCH_CHECK(ssp.standardise == sig.standardisation.standardise);
    CATCH_CHECK(ssp.mean == sig.standardisation.mean);
    CATCH_CHECK(ssp.stdev == sig.standardisation.stdev);

    ConvParams conv1 = config.convs[0];
    CATCH_CHECK(conv1.activation == Activation::SWISH);
    CATCH_CHECK(conv1.insize == 1);
    CATCH_CHECK(conv1.size == 4);  // default first_conv value
    CATCH_CHECK(conv1.stride == 1);
    CATCH_CHECK(conv1.winlen == 5);

    ConvParams conv2 = config.convs[1];
    CATCH_CHECK(conv2.activation == Activation::SWISH);
    CATCH_CHECK(conv2.insize == 4);
    CATCH_CHECK(conv2.size == 16);
    CATCH_CHECK(conv2.stride == 1);
    CATCH_CHECK(conv2.winlen == 5);

    ConvParams conv3 = config.convs[2];
    CATCH_CHECK(conv3.activation == Activation::SWISH);
    CATCH_CHECK(conv3.insize == 16);
    CATCH_CHECK(conv3.size == 768);
    CATCH_CHECK(conv3.stride == 5);
    CATCH_CHECK(conv3.winlen == 19);
}

CATCH_TEST_CASE(CUT_TAG ": test sample_type", CUT_TAG) {
    CATCH_SECTION("test sample type dna") {
        const fs::path path =
                fs::path(get_data_dir("model_configs/sample_type_d_e8.2_400bps_sup@v5.0.0"));
        const BasecallModelConfig config = load_model_config(path);

        CATCH_CHECK(config.sample_type == dorado::models::SampleType::DNA);
    }
    CATCH_SECTION("test sample type rna004") {
        const fs::path path = fs::path(get_data_dir("model_configs/sample_type_130bps_sup@v3.0.1"));
        const BasecallModelConfig config = load_model_config(path);

        CATCH_CHECK(config.sample_type == dorado::models::SampleType::RNA004);
    }
}