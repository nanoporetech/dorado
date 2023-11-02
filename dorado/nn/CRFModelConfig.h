#pragma once

#include "utils/math_utils.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado {

enum class Activation { SWISH, SWISH_CLAMP, TANH };
std::string to_string(const Activation& activation);

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
std::string to_string(const ScalingStrategy& strategy);
ScalingStrategy scaling_strategy_from_string(const std::string& strategy);

struct SignalNormalisationParams {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;
    ScalingStrategy strategy = ScalingStrategy::QUANTILE;

    std::string to_string() const {
        std::string str = "SignalNormalisationParams {";
        str += " strategy:" + dorado::to_string(strategy);
        if (strategy == ScalingStrategy::QUANTILE) {
            str += " quantile_a:" + std::to_string(quantile_a);
            str += " quantile_b:" + std::to_string(quantile_b);
            str += " shift_multiplier:" + std::to_string(shift_multiplier);
            str += " scale_multiplier:" + std::to_string(scale_multiplier);
        }
        str += "}";
        return str;
    }
};

struct ConvParams {
    int insize;
    int size;
    int winlen;
    int stride = 1;
    Activation activation;

    std::string to_string() const {
        std::string str = "ConvParams {";
        str += " insize:" + std::to_string(insize);
        str += " size:" + std::to_string(size);
        str += " winlen:" + std::to_string(winlen);
        str += " stride:" + std::to_string(stride);
        str += " activation:" + dorado::to_string(activation);
        str += "}";
        return str;
    };
};

enum SampleType {
    DNA,
    RNA002,
    RNA004,
};

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale = 1.0f;
    float qbias = 0.0f;
    int lstm_size = 0;
    int stride = 1;
    bool bias = true;
    bool clamp = false;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    std::optional<int> out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    float scale = 1.0f;
    int num_features;
    int sample_rate = -1;
    SignalNormalisationParams signal_norm_params;
    std::filesystem::path model_path;

    // Start position for mean Q-score calculation for
    // short reads.
    int32_t mean_qscore_start_pos = -1;

    SampleType sample_type;

    // convolution layer params
    std::vector<ConvParams> convs;

    std::string to_string() const {
        std::string str = "CRFModelConfig {";
        str += " qscale:" + std::to_string(qscale);
        str += " qbias:" + std::to_string(qbias);
        str += " stride:" + std::to_string(stride);
        str += " bias:" + std::to_string(bias);
        str += " clamp:" + std::to_string(clamp);
        str += " out_features:" + std::to_string(out_features.value_or(-1));
        str += " state_len:" + std::to_string(state_len);
        str += " outsize:" + std::to_string(outsize);
        str += " blank_score:" + std::to_string(blank_score);
        str += " scale:" + std::to_string(scale);
        str += " num_features:" + std::to_string(num_features);
        str += " sample_rate:" + std::to_string(sample_rate);
        str += " mean_qscore_start_pos:" + std::to_string(mean_qscore_start_pos);
        str += " signal_norm_params:" + signal_norm_params.to_string();
        str += " convs: {";
        for (int c = 0; c < convs.size(); c++) {
            str += " " + std::to_string(c) + ": " + convs[c].to_string();
        }
        str += "}";
        return str;
    };
};

CRFModelConfig load_crf_model_config(const std::filesystem::path& path);

uint16_t get_model_sample_rate(const std::filesystem::path& model_path);

inline bool sample_rates_compatible(uint16_t data_sample_rate, uint16_t model_sample_rate) {
    return utils::eq_with_tolerance(data_sample_rate, model_sample_rate,
                                    static_cast<uint16_t>(100));
}

int32_t get_model_mean_qscore_start_pos(const CRFModelConfig& model_config);

bool is_rna_model(const CRFModelConfig& model_config);

}  // namespace dorado
