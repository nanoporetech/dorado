#pragma once

#include <torch/torch.h>

void serialise_tensor(torch::Tensor t, const std::string& path);
std::vector<torch::Tensor> load_weights(const std::string& dir);
