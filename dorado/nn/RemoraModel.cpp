#include "RemoraModel.h"

#include "utils/module_utils.h"
#include "utils/tensor_utils.h"

#include <toml.hpp>
#include <torch/torch.h>

#include <stdexcept>

using namespace torch::nn;
using namespace torch::indexing;

namespace {
template <class Model>
ModuleHolder<AnyModule> populate_model(Model&& model,
                                       const std::filesystem::path& path,
                                       torch::TensorOptions options) {
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}
}  // namespace

namespace dorado {

namespace nn {

struct UnpaddedConvolutionImpl : Module {
    UnpaddedConvolutionImpl(int size = 1, int outsize = 1, int k = 1, int stride = 1) {
        conv = register_module("conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) { return activation(conv(x)); }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
};

TORCH_MODULE(UnpaddedConvolution);

struct RemoraConvModelImpl : Module {
    RemoraConvModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", UnpaddedConvolution(1, 4, 11, 1));
        sig_conv2 = register_module("sig_conv2", UnpaddedConvolution(4, 16, 11, 1));
        sig_conv3 = register_module("sig_conv3", UnpaddedConvolution(16, size, 9, 3));

        seq_conv1 = register_module("seq_conv1", UnpaddedConvolution(kmer_len * 4, 16, 11, 1));
        seq_conv2 = register_module("seq_conv2", UnpaddedConvolution(16, 32, 11, 1));
        seq_conv3 = register_module("seq_conv3", UnpaddedConvolution(32, size, 9, 3));

        merge_conv1 = register_module("merge_conv1", UnpaddedConvolution(size * 2, size, 5, 1));
        merge_conv2 = register_module("merge_conv2", UnpaddedConvolution(size, size, 5, 1));
        merge_conv3 = register_module("merge_conv3", UnpaddedConvolution(size, size, 3, 2));
        merge_conv4 = register_module("merge_conv4", UnpaddedConvolution(size, size, 3, 2));

        linear = register_module("linear", Linear(size * 3, num_out));
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
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

        z = z.softmax(1);

        return z;
    }

    void load_state_dict(const std::vector<torch::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    UnpaddedConvolution sig_conv1{nullptr};
    UnpaddedConvolution sig_conv2{nullptr};
    UnpaddedConvolution sig_conv3{nullptr};
    UnpaddedConvolution seq_conv1{nullptr};
    UnpaddedConvolution seq_conv2{nullptr};
    UnpaddedConvolution seq_conv3{nullptr};
    UnpaddedConvolution merge_conv1{nullptr};
    UnpaddedConvolution merge_conv2{nullptr};
    UnpaddedConvolution merge_conv3{nullptr};
    UnpaddedConvolution merge_conv4{nullptr};
    Linear linear{nullptr};
};

const std::vector<std::string> RemoraConvModelImpl::weight_tensors{
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

struct RemoraConvLSTMModelImpl : Module {
    RemoraConvLSTMModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", UnpaddedConvolution(1, 4, 5, 1));
        sig_conv2 = register_module("sig_conv2", UnpaddedConvolution(4, 16, 5, 1));
        sig_conv3 = register_module("sig_conv3", UnpaddedConvolution(16, size, 9, 3));

        seq_conv1 = register_module("seq_conv1", UnpaddedConvolution(kmer_len * 4, 16, 5, 1));
        seq_conv2 = register_module("seq_conv2", UnpaddedConvolution(16, size, 13, 3));

        merge_conv1 = register_module("merge_conv1", UnpaddedConvolution(size * 2, size, 5, 1));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(size, size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(size, size)));

        linear = register_module("linear", Linear(size, num_out));

        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
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

        auto z = torch::cat({sigs, seqs}, 1);
        z = merge_conv1(z);
        z = z.permute({2, 0, 1});

        auto [z1, h1] = lstm1(z);
        z1 = activation(z1);

        z1 = z1.flip(0);
        auto [z2, h2] = lstm2(z1);
        z2 = activation(z2);
        z2 = z2.flip(0);

        z = z2.index({-1}).permute({0, 1});
        z = linear(z);
        z = z.softmax(1);

        return z;
    }

    void load_state_dict(std::vector<torch::Tensor> weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    UnpaddedConvolution sig_conv1{nullptr};
    UnpaddedConvolution sig_conv2{nullptr};
    UnpaddedConvolution sig_conv3{nullptr};
    UnpaddedConvolution seq_conv1{nullptr};
    UnpaddedConvolution seq_conv2{nullptr};
    UnpaddedConvolution merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
};

const std::vector<std::string> RemoraConvLSTMModelImpl::weight_tensors{
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

TORCH_MODULE(RemoraConvModel);
TORCH_MODULE(RemoraConvLSTMModel);

}  // namespace nn

ModuleHolder<AnyModule> load_remora_model(const std::filesystem::path& model_path,
                                          torch::TensorOptions options) {
    auto config = toml::parse(model_path / "config.toml");

    const auto& general_params = toml::find(config, "general");
    const auto model_type = toml::find<std::string>(general_params, "model");

    const auto& model_params = toml::find(config, "model_params");
    const auto size = toml::find<int>(model_params, "size");
    const auto kmer_len = toml::find<int>(model_params, "kmer_len");
    const auto num_out = toml::find<int>(model_params, "num_out");

    if (model_type == "conv_lstm") {
        auto model = nn::RemoraConvLSTMModel(size, kmer_len, num_out);
        return populate_model(model, model_path, options);
    }

    if (model_type == "conv_only") {
        auto model = nn::RemoraConvModel(size, kmer_len, num_out);
        return populate_model(model, model_path, options);
    }

    throw std::runtime_error("Unknown model type in config file.");
}

}  // namespace dorado
