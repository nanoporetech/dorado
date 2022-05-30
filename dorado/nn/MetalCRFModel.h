#pragma once

#include <torch/torch.h>

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_mtl_model(const std::string& path,
                                                                 int batch_size,
                                                                 int chunk_size,
                                                                 torch::TensorOptions options);
