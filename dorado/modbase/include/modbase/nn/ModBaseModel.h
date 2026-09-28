#pragma once

#include "config/ModBaseModelConfigFwd.h"
#include "torch_utils/module_utils.h"

#include <torch/nn.h>

#include <optional>
#include <tuple>
#include <vector>

namespace dorado::modbase {

using ModBaseLatentSpaceModels =
        std::tuple<torch::nn::ModuleHolder<torch::nn::AnyModule>,
                   torch::nn::ModuleHolder<torch::nn::AnyModule>,
                   std::vector<torch::nn::ModuleHolder<torch::nn::AnyModule>>>;

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const config::ModBaseModelConfig& config,
        const at::TensorOptions& options,
        const int batchsize);

// Returns the sequence model, common model, and classifier models in config order, or std::nullopt
// when configs is empty.
std::optional<ModBaseLatentSpaceModels> load_modbase_latent_space_model(
        const std::vector<config::ModBaseModelConfig>& configs,
        const at::TensorOptions& options);

}  // namespace dorado::modbase
