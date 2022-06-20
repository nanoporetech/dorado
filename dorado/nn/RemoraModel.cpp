#include "RemoraModel.h"

#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#include <toml.hpp>
#include <torch/torch.h>

using namespace torch::nn;

struct ConvBatchNormImpl : Module {
    ConvBatchNormImpl(int size = 1,
                      int outsize = 1,
                      int k = 1,
                      int stride = 1,
                      int num_features = 1) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        batch_norm = register_module("batch_norm", BatchNorm1d(num_features));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) { return activation(batch_norm(conv(x))); }

    Conv1d conv{nullptr};
    BatchNorm1d batch_norm{nullptr};
    SiLU activation{nullptr};
};

TORCH_MODULE(ConvBatchNorm);

struct RemoraConvModelImpl : Module {
    RemoraConvModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", ConvBatchNorm(1, 4, 11, 1, 4));
        sig_conv2 = register_module("sig_conv2", ConvBatchNorm(4, 16, 11, 1, 16));
        sig_conv3 = register_module("sig_conv3", ConvBatchNorm(16, size, 9, 3, size));

        seq_conv1 = register_module("seq_conv1", ConvBatchNorm(kmer_len * 4, 16, 11, 1, 16));
        seq_conv2 = register_module("seq_conv2", ConvBatchNorm(16, 32, 11, 1, 32));
        seq_conv3 = register_module("seq_conv3", ConvBatchNorm(32, size, 9, 3, size));

        merge_conv1 = register_module("merge_conv1", ConvBatchNorm(size * 2, size, 5, 1, size));
        merge_conv2 = register_module("merge_conv2", ConvBatchNorm(size, size, 5, 1, size));
        merge_conv3 = register_module("merge_conv3", ConvBatchNorm(size, size, 3, 2, size));
        merge_conv4 = register_module("merge_conv4", ConvBatchNorm(size, size, 3, 2, size));

        linear = register_module("linear", Linear(size * 3, num_out));
    }

    void load_state_dict(const std::vector<torch::Tensor>& weights) {
        ::utils::load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

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

    std::vector<torch::Tensor> load_weights(const std::string& dir) {
        auto weights = std::vector<torch::Tensor>();
        auto tensors = std::vector<std::string>{};  // TODO!

        return ::utils::load_weights(dir, tensors);
    }

    ConvBatchNorm sig_conv1{nullptr};
    ConvBatchNorm sig_conv2{nullptr};
    ConvBatchNorm sig_conv3{nullptr};
    ConvBatchNorm seq_conv1{nullptr};
    ConvBatchNorm seq_conv2{nullptr};
    ConvBatchNorm seq_conv3{nullptr};
    ConvBatchNorm merge_conv1{nullptr};
    ConvBatchNorm merge_conv2{nullptr};
    ConvBatchNorm merge_conv3{nullptr};
    ConvBatchNorm merge_conv4{nullptr};
    Linear linear{nullptr};
};

struct RemoraConvLSTMModelImpl : Module {
    RemoraConvLSTMModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", ConvBatchNorm(1, 4, 5, 1, 4));
        sig_conv2 = register_module("sig_conv2", ConvBatchNorm(4, 16, 5, 1, 16));
        sig_conv3 = register_module("sig_conv3", ConvBatchNorm(16, size, 9, 3, size));

        seq_conv1 = register_module("seq_conv1", ConvBatchNorm(kmer_len * 4, 16, 5, 1, 16));
        seq_conv2 = register_module("seq_conv2", ConvBatchNorm(16, size, 13, 3, size));

        merge_conv1 = register_module("merge_conv1", ConvBatchNorm(size * 2, size, 5, 1, size));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(size, size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(size, size)));

        linear = register_module("linear", Linear(size, num_out));

        activation = register_module("activation", SiLU());
    }

    void load_state_dict(std::vector<torch::Tensor> weights) {
        ::utils::load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

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

        z = z2.index({-1});
        z = linear(z);

        return z;
    }

    std::vector<torch::Tensor> load_weights(const std::string& dir) {
        auto weights = std::vector<torch::Tensor>();
        auto tensors = std::vector<std::string>{
                "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
                "sig_bn1.weight.tensor",     "sig_bn1.bias.tensor",

                "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
                "sig_bn2.weight.tensor",     "sig_bn2.bias.tensor",

                "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",
                "sig_bn3.weight.tensor",     "sig_bn3.bias.tensor",

                "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
                "seq_bn1.weight.tensor",     "seq_bn1.bias.tensor",

                "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
                "seq_bn2.weight.tensor",     "seq_bn2.bias.tensor",

                "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
                "merge_bn.weight.tensor",    "merge_bn.bias.tensor",

                "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
                "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

                "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
                "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

                "fc.weight.tensor",          "fc.bias.tensor",
        };

        return ::utils::load_weights(dir, tensors);
    }

    ConvBatchNorm sig_conv1{nullptr};
    ConvBatchNorm sig_conv2{nullptr};
    ConvBatchNorm sig_conv3{nullptr};
    ConvBatchNorm seq_conv1{nullptr};
    ConvBatchNorm seq_conv2{nullptr};
    ConvBatchNorm merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
    Softmax softmax{nullptr};
};
TORCH_MODULE(RemoraConvModel);
TORCH_MODULE(RemoraConvLSTMModel);

ModuleHolder<AnyModule> load_remora_model(const std::string& path, torch::TensorOptions options) {
    auto config = toml::parse(path + "/config.toml");

    const auto& model_params = toml::find(config, "model_params");
    const auto size = toml::find<int>(model_params, "size");
    const auto kmer_len = toml::find<int>(model_params, "kmer_len");
    const auto num_out = toml::find<int>(model_params, "num_out");

    // TODO: detect the correct model!
    auto model = RemoraConvLSTMModel(size, kmer_len, num_out);
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    // model->to(options.dtype_opt().value().toScalarType()); // ?
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);

    return holder;
}

RemoraCaller::RemoraCaller(const std::string& model, std::string device) {
    m_options = torch::TensorOptions().dtype(dtype).device(device == "metal" ? "cpu" : device);
    m_module = load_remora_model(model, m_options);

    auto config = toml::parse(model + "/config.toml");
    const auto& params = toml::find(config, "modbases");
    const auto num_motifs = toml::find<int>(params, "num_motifs");
    std::vector<std::string> mod_long_names;
    std::vector<std::string> motifs;
    std::vector<int> motif_offsets;
    for (auto i = 0; i < num_motifs; ++i) {
        auto counter_string = std::to_string(i);
        mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + counter_string));
        motifs.push_back(toml::find<std::string>(params, "motif_" + counter_string));
        motif_offsets.push_back(toml::find<int>(params, "motif_offset_" + counter_string));
    }

    std::pair<int, int> chunk_contexts{toml::find<int>(params, "chunk_context_0"),
                                       toml::find<int>(params, "chunk_context_1")};

    std::pair<int, int> kmer_context_bases{toml::find<int>(params, "kmer_context_bases_0"),
                                           toml::find<int>(params, "kmer_context_bases_1")};

    const auto offset = toml::find<int>(params, "offset");
    const auto mod_bases = toml::find<std::string>(params, "mod_bases");
}

RemoraRunner::RemoraRunner(const std::vector<std::string>& model_paths, std::string device) {
    // no metal implementation yet, force to cpu
    for (const auto& model : model_paths) {
        m_callers.push_back(std::make_shared<RemoraCaller>(model, device));
    }
}
