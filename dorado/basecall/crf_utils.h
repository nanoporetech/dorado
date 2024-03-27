#pragma once

#include <torch/nn.h>

#include <filesystem>
#include <vector>

namespace dorado::basecall {

struct CRFModelConfig;

std::vector<at::Tensor> load_crf_model_weights(const std::filesystem::path& dir,
                                               bool decomposition,
                                               bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const CRFModelConfig& model_config,
                                                             const at::TensorOptions& options);

std::vector<at::Tensor> load_tx_model_weights(const std::filesystem::path& dir);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_tx_model(const CRFModelConfig& model_config,
                                                            const at::TensorOptions& options);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_model(const CRFModelConfig& model_config,
                                                         const torch::TensorOptions& options);

size_t auto_calculate_num_runners(const CRFModelConfig& model_config,
                                  size_t batch_size,
                                  float memory_fraction);

}  // namespace dorado::basecall
