#pragma once

#include <torch/torch.h>

#include <filesystem>

namespace dorado {

torch::nn::ModuleHolder<torch::nn::AnyModule> load_remora_model(
        const std::filesystem::path& model_path,
        torch::TensorOptions options);

}  // namespace dorado
