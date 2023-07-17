#pragma once

#include "utils/math_utils.h"

#include <torch/torch.h>

#include <filesystem>
#include <optional>
#include <vector>

namespace dorado {

struct SignalNormalisationParams {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;
    bool quantile_scaling = true;
};

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale = 1.0f;
    float qbias = 0.0f;
    int conv = 4;
    int insize = 0;
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
};

CRFModelConfig load_crf_model_config(const std::filesystem::path& path);

std::vector<torch::Tensor> load_crf_model_weights(const std::filesystem::path& dir,
                                                  bool decomposition,
                                                  bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const CRFModelConfig& model_config,
                                                             const torch::TensorOptions& options);

uint16_t get_model_sample_rate(const std::filesystem::path& model_path);

inline bool sample_rates_compatible(uint16_t data_sample_rate, uint16_t model_sample_rate) {
    return utils::eq_with_tolerance(data_sample_rate, model_sample_rate,
                                    static_cast<uint16_t>(100));
}

int32_t get_model_mean_qscore_start_pos(const CRFModelConfig& model_config);

}  // namespace dorado
