#pragma once

#include <torch/torch.h>

#include <string>
#include <tuple>

std::tuple<torch::nn::ModuleHolder<torch::nn::AnyModule>, size_t> load_crf_model(
        const std::string& path,
        int batch_size,
        int chunk_size,
        torch::TensorOptions options);
