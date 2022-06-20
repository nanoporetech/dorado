#pragma once

#include <torch/torch.h>

#include <string>

torch::nn::ModuleHolder<torch::nn::AnyModule> load_remora_model(const std::string& path,
                                                                torch::TensorOptions options);
