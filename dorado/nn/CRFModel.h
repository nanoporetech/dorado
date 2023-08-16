#pragma once

#include "utils/math_utils.h"

#include <torch/torch.h>

#include <filesystem>
#include <optional>
#include <vector>

namespace dorado {

struct CRFModelConfig;

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
