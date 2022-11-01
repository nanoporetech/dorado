#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <tuple>

namespace dorado {

std::tuple<torch::nn::ModuleHolder<torch::nn::AnyModule>, size_t> load_crf_model(
        const std::filesystem::path& path,
        int batch_size,
        int chunk_size,
        torch::TensorOptions options);

}  // namespace dorado
