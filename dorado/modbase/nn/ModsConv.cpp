#include "ModsConv.h"

#include "torch_utils/gpu_profiling.h"

#include <stdexcept>

namespace dorado::modbase::model {

ModsConvImpl::ModsConvImpl(int size, int outsize, int k, int stride, int padding)
        : name("conv_act_" + std::to_string(size)) {
    conv = register_module(
            "conv",
            torch::nn::Conv1d(
                    torch::nn::Conv1dOptions(size, outsize, k).stride(stride).padding(padding)));
    activation = config::Activation::SWISH;
}

ModsConvImpl::ModsConvImpl(const config::ConvParams& params)
        : name("conv_act_" + std::to_string(params.size)) {
    conv = register_module(
            "conv",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(params.insize, params.size, params.winlen)
                                      .stride(params.stride)
                                      .padding(params.winlen / 2)));
    activation = params.activation;
}

at::Tensor ModsConvImpl::forward(const at::Tensor& x) {
    utils::ScopedProfileRange spr(name.c_str(), 3);
    at::Tensor result = conv(x);

    switch (activation) {
    case config::Activation::SWISH:
        return at::silu(result);
    case config::Activation::SWISH_CLAMP:
        throw std::runtime_error("ModsConv is not implemented for SWISH_CLAMP");
    case config::Activation::TANH:
        return at::tanh(result);
    }
    throw std::logic_error("ModsConv has unsupported activation");
}

}  // namespace dorado::modbase::model
