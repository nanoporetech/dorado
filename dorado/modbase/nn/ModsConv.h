#pragma once

#include "config/common.h"

#include <torch/nn.h>

#include <string>
#include <vector>

namespace dorado::modbase::model {

struct ModsConvImpl : torch::nn::Module {
    ModsConvImpl(int size, int outsize, int k, int stride, int padding);
    explicit ModsConvImpl(const config::ConvParams& params);

    at::Tensor forward(const at::Tensor& x);

    const std::string name;
    torch::nn::Conv1d conv{nullptr};
    config::Activation activation;
};

TORCH_MODULE(ModsConv);

}  // namespace dorado::modbase::model
