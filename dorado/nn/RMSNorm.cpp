#include "nn/RMSNorm.h"

#include <c10/core/TensorOptions.h>
#include <spdlog/spdlog.h>
#include <torch/types.h>

namespace dorado::nn {

RMSNormImpl::RMSNormImpl(int hidden_size_) : hidden_size(hidden_size_) {
    weight = at::ones({hidden_size});
    register_parameter("weight", weight, false);
}

at::Tensor RMSNormImpl::forward(at::Tensor x) {
    at::Tensor rstd = torch::rsqrt(x.square().mean(-1, true).add_(eps));
    x.mul_(rstd).mul_(weight);
    return x;
}

}  // namespace dorado::nn
