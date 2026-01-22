#pragma once

#include <ATen/ATen.h>
#include <torch/nn/modules/container/modulelist.h>

namespace dorado::nn {

struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(int hidden_size_);
    at::Tensor forward(at::Tensor x);

    at::Tensor weight;
    const int hidden_size;
    const float eps{1e-5f};
};

TORCH_MODULE(RMSNorm);

}  // namespace dorado::nn
