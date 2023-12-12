#pragma once

#include <ATen/core/TensorBody.h>

#include <filesystem>
#include <vector>

namespace dorado::basecall {

struct CRFModelConfig;

std::vector<at::Tensor> load_crf_model_weights(const std::filesystem::path& dir,
                                               bool decomposition,
                                               bool bias);

size_t auto_calculate_num_runners(const CRFModelConfig& model_config,
                                  size_t batch_size,
                                  float memory_fraction);

}  // namespace dorado::basecall
