#pragma once

#include "torch_utils/module_utils.h"

#include <torch/nn.h>

namespace dorado::config {
struct ModBaseModelConfig;
}

namespace dorado::modbase {

dorado::utils::ModuleWrapper load_modbase_model(const config::ModBaseModelConfig& config,
                                                const at::TensorOptions& options);

std::vector<float> load_kmer_refinement_levels(const config::ModBaseModelConfig& config);

}  // namespace dorado::modbase
