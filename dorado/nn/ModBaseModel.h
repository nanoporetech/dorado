#pragma once

#include <torch/nn.h>

#include <filesystem>

namespace dorado {

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const std::filesystem::path& model_path,
        at::TensorOptions options);

}  // namespace dorado
