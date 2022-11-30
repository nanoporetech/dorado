#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <optional>
#include <vector>

namespace dorado {

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale;
    float qbias;
    int conv;
    int insize;
    int stride;
    bool bias;
    bool clamp;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    std::optional<int> out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    float scale;
    int num_features;
};

CRFModelConfig load_crf_model_config(const std::filesystem::path& path);

std::vector<torch::Tensor> load_crf_model_weights(const std::filesystem::path& dir,
                                                  bool decomposition,
                                                  bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const std::filesystem::path& path,
                                                             const CRFModelConfig& model_config,
                                                             int batch_size,
                                                             int chunk_size,
                                                             const torch::TensorOptions& options);

}  // namespace dorado
