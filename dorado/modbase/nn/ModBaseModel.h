#pragma once

#include "config/ModBaseModelConfigFwd.h"
#include "torch_utils/module_utils.h"

#include <torch/nn.h>

namespace dorado::modbase {

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const config::ModBaseModelConfig& config,
        const at::TensorOptions& options,
        const int batchsize);

}  // namespace dorado::modbase
