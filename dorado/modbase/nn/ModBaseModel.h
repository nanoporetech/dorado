#pragma once

#include "modbase/ModBaseModelConfig.h"

#include <torch/nn.h>

#include <filesystem>

namespace dorado::modbase {

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(const ModBaseModelConfig& config,
                                                                 const at::TensorOptions& options);

}  // namespace dorado::modbase
