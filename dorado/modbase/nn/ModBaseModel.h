#pragma once

#include "modbase/ModBaseModelConfig.h"
#include "torch_utils/module_utils.h"

#include <torch/nn.h>

#include <filesystem>

namespace dorado::modbase {

dorado::utils::ModuleWrapper load_modbase_model(const ModBaseModelConfig& config,
                                                const at::TensorOptions& options);

}  // namespace dorado::modbase
