#include "ModBaseConvModel.h"

#include "torch_utils/module_utils.h"

namespace dorado::modbase::model {

ModBaseConvModelImpl::ModBaseConvModelImpl(const config::modbase::ConvOnlyModelConfig& config) {
    const auto& params = config.general;
    const auto size = params.size;
    const auto kmer_len = params.kmer_len;
    const auto num_out = params.num_out;
    sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 11, 1, 0));
    sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 11, 1, 0));
    sig_conv3 = register_module("sig_conv3", ModsConv(16, size, 9, 3, 0));

    seq_conv1 = register_module("seq_conv1", ModsConv(kmer_len * 4, 16, 11, 1, 0));
    seq_conv2 = register_module("seq_conv2", ModsConv(16, 32, 11, 1, 0));
    seq_conv3 = register_module("seq_conv3", ModsConv(32, size, 9, 3, 0));

    merge_conv1 = register_module("merge_conv1", ModsConv(size * 2, size, 5, 1, 0));
    merge_conv2 = register_module("merge_conv2", ModsConv(size, size, 5, 1, 0));
    merge_conv3 = register_module("merge_conv3", ModsConv(size, size, 3, 2, 0));
    merge_conv4 = register_module("merge_conv4", ModsConv(size, size, 3, 2, 0));

    linear = register_module("linear", torch::nn::Linear(size * 3, num_out));
}

at::Tensor ModBaseConvModelImpl::forward(at::Tensor sigs, at::Tensor seqs) {
    sigs = sig_conv1(sigs);
    sigs = sig_conv2(sigs);
    sigs = sig_conv3(sigs);

    // We are supplied one hot encoded sequences as (batch, signal, kmer_len * base_count) int8.
    // We need (batch, kmer_len * base_count, signal) and a dtype compatible with the float16
    // weights.
    const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;
    seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
    seqs = seq_conv1(seqs);
    seqs = seq_conv2(seqs);
    seqs = seq_conv3(seqs);

    auto z = torch::cat({sigs, seqs}, 1);

    z = merge_conv1(z);
    z = merge_conv2(z);
    z = merge_conv3(z);
    z = merge_conv4(z);

    z = z.flatten(1);
    z = linear(z);

    return z.softmax(1);
}

void ModBaseConvModelImpl::load_state_dict(const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseConvModelImpl::load_weights(const std::filesystem::path& dir) {
    return utils::load_tensors(dir, weight_tensors());
}

std::vector<std::string> ModBaseConvModelImpl::weight_tensors() {
    return {
            "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
            "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
            "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

            "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
            "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
            "seq_conv3.weight.tensor",   "seq_conv3.bias.tensor",

            "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
            "merge_conv2.weight.tensor", "merge_conv2.bias.tensor",
            "merge_conv3.weight.tensor", "merge_conv3.bias.tensor",
            "merge_conv4.weight.tensor", "merge_conv4.bias.tensor",

            "fc.weight.tensor",          "fc.bias.tensor",
    };
}

}  // namespace dorado::modbase::model
