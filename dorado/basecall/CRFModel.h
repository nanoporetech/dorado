#pragma once

#include <torch/nn.h>

namespace dorado::basecall {

struct CRFModelConfig;

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const CRFModelConfig& model_config,
                                                             const at::TensorOptions& options);

}  // namespace dorado::basecall
