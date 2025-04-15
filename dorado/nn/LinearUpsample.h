#pragma once

#include "config/common.h"

#include <torch/nn.h>

namespace dorado::nn {

struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(const int size, const int scale_factor);
    LinearUpsampleImpl(const config::LinearUpsampleParams &params);

    at::Tensor forward(const at::Tensor &x);

    const int scale_factor;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearUpsample);

}  // namespace dorado::nn