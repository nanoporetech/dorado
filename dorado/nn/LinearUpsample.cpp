#include "LinearUpsample.h"

#include "config/common.h"

namespace dorado::nn {

LinearUpsampleImpl::LinearUpsampleImpl(const int size, const int scale_factor_)
        : scale_factor(scale_factor_) {
    linear = register_module(
            "linear",
            torch::nn::Linear(torch::nn::LinearOptions(size, scale_factor * size).bias(true)));
};

LinearUpsampleImpl::LinearUpsampleImpl(const config::LinearUpsampleParams &params)
        : LinearUpsampleImpl(params.size, params.scale_factor) {};

at::Tensor LinearUpsampleImpl::forward(const at::Tensor &x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    at::Tensor out = linear(x).reshape({N, scale_factor * T, C});
    return out;
};

}  // namespace dorado::nn