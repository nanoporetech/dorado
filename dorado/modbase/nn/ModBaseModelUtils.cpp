#include "ModBaseModelUtils.h"

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

std::vector<std::string> load_modbase_conv_lstm_weights(bool has_upsample) {
    std::vector<std::string> tensors{
            "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
            "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
            "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

            "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
            "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",

            "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",

            "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
            "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

            "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
            "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

            "fc.weight.tensor",          "fc.bias.tensor",
    };

    if (has_upsample) {
        tensors.push_back("linear_up.linear.weight.tensor");
        tensors.push_back("linear_up.linear.bias.tensor");
    }
    return tensors;
}

}  // namespace dorado::modbase::model
