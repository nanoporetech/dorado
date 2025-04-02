#include "ModBaseModel.h"

#include "config/ModBaseModelConfig.h"
#include "spdlog/spdlog.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/dev_utils.h"

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

#include <stdexcept>
#include <string>
#include <vector>

using namespace torch::nn;
using namespace torch::indexing;

namespace {

template <class Model>
ModuleHolder<AnyModule> populate_model(Model&& model,
                                       const std::filesystem::path& path,
                                       const at::TensorOptions& options) {
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(std::move(model));
    auto holder = ModuleHolder<AnyModule>(std::move(module));
    return holder;
}
}  // namespace

namespace dorado::modbase {

namespace nn {

struct ModsConvImpl : Module {
    ModsConvImpl(int size, int outsize, int k, int stride, int padding)
            : name("conv_act_" + std::to_string(size)) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(padding)));
        activation = config::Activation::SWISH;
    }

    ModsConvImpl(const config::ConvParams& params)
            : name("conv_act_" + std::to_string(params.size)) {
        spdlog::debug("{} {}", name, params.to_string());
        conv = register_module("conv",
                               Conv1d(Conv1dOptions(params.insize, params.size, params.winlen)
                                              .stride(params.stride)
                                              .padding(params.winlen / 2)));
        activation = params.activation;
    }

    at::Tensor forward(const at::Tensor& x) {
        utils::ScopedProfileRange spr(name.c_str(), 3);
        at::Tensor x_ = conv(x);

        switch (activation) {
        case config::Activation::SWISH:
            return at::silu(x_);
        case config::Activation::SWISH_CLAMP:
            throw std::runtime_error("ModsConv is not implemented for SWISH_CLAMP");
        case config::Activation::TANH:
            return at::tanh(x_);
        }
        throw std::logic_error("ModsConv has unsupported activation");
    }

    const std::string name;
    Conv1d conv{nullptr};
    config::Activation activation;
};

TORCH_MODULE(ModsConv);

struct ModBaseConvModelImpl : Module {
    ModBaseConvModelImpl(int size, int kmer_len, int num_out) {
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

        linear = register_module("linear", Linear(size * 3, num_out));
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
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

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv seq_conv3{nullptr};
    ModsConv merge_conv1{nullptr};
    ModsConv merge_conv2{nullptr};
    ModsConv merge_conv3{nullptr};
    ModsConv merge_conv4{nullptr};
    Linear linear{nullptr};
};

const std::vector<std::string> ModBaseConvModelImpl::weight_tensors{
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

struct ModBaseConvLSTMModelImpl : Module {
    ModBaseConvLSTMModelImpl(int size, int kmer_len, int num_out, bool is_conv_lstm_v2, int stride)
            : m_is_conv_lstm_v2(is_conv_lstm_v2) {
        // conv_lstm_v2 models are padded to ensure the output shape is nicely indexable by the stride
        sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv3 = register_module("sig_conv3",
                                    ModsConv(16, size, 9, stride, m_is_conv_lstm_v2 ? 4 : 0));

        seq_conv1 = register_module("seq_conv1",
                                    ModsConv(kmer_len * 4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        seq_conv2 = register_module("seq_conv2",
                                    ModsConv(16, size, 13, stride, m_is_conv_lstm_v2 ? 6 : 0));

        merge_conv1 = register_module("merge_conv1",
                                      ModsConv(size * 2, size, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(size, size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(size, size)));

        linear = register_module("linear", Linear(size, num_out));

        activation = register_module("activation", SiLU());
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        // INPUT sigs: NCT & seqs: NTC
        {
            utils::ScopedProfileRange spr("sig convs", 2);
            sigs = sig_conv1(sigs);
            sigs = sig_conv2(sigs);
            sigs = sig_conv3(sigs);
        }

        // We are supplied one hot encoded sequences as (batch, signal, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

        // seqs: NTC -> NCT
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        {
            utils::ScopedProfileRange spr("seq convs", 2);
            seqs = seq_conv1(seqs);
            seqs = seq_conv2(seqs);
        }

        // z: NCT
        auto z = torch::cat({sigs, seqs}, 1);
        {
            utils::ScopedProfileRange spr("merge convs", 2);
            // z: NCT -> TNC
            z = merge_conv1(z).permute({2, 0, 1});
        }
        {
            utils::ScopedProfileRange spr("lstm1", 2);
            auto [z1, h1] = lstm1(z);
            z = activation(z1).flip(0);
        }
        {
            utils::ScopedProfileRange spr("lstm2", 2);
            auto [z2, h2] = lstm2(z);
            z = activation(z2).flip(0);
        }

        if (m_is_conv_lstm_v2) {
            utils::ScopedProfileRange spr("chunk_linear", 2);
            // TNC -> NTC
            z = linear(z.permute({1, 0, 2})).softmax(2).flatten(1);
        } else {
            utils::ScopedProfileRange spr("context_linear", 2);
            // Take the final time step: TNC -> tNC -> NC
            z = z.index({-1}).permute({0, 1});
            z = linear(z).softmax(1);
        }
        return z;
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    const bool m_is_conv_lstm_v2{false};
    static const std::vector<std::string> weight_tensors;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
};

const std::vector<std::string> ModBaseConvLSTMModelImpl::weight_tensors{
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

struct ModBaseConvLSTMV3ModelImpl : Module {
    ModBaseConvLSTMV3ModelImpl(const config::ModBaseModelConfig& config) : m_config(config) {
        const auto& sig_cvs = m_config.general.modules->signal_convs;
        if (sig_cvs.size() != 3) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
        }
        sig_conv1 = register_module("sig_conv1", ModsConv(sig_cvs.at(0)));
        sig_conv2 = register_module("sig_conv2", ModsConv(sig_cvs.at(1)));
        sig_conv3 = register_module("sig_conv3", ModsConv(sig_cvs.at(2)));

        const auto& seq_cvs = m_config.general.modules->sequence_convs;
        if (seq_cvs.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
        }
        seq_conv1 = register_module("seq_conv1", ModsConv(seq_cvs.at(0)));
        seq_conv2 = register_module("seq_conv2", ModsConv(seq_cvs.at(1)));

        merge_conv1 =
                register_module("merge_conv1", ModsConv(m_config.general.modules->merge_conv));

        const auto& lstms = m_config.general.modules->lstms;
        if (lstms.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
        }
        lstm1 = register_module("lstm1", LSTM(LSTMOptions(lstms.at(0).size, lstms.at(0).size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(lstms.at(1).size, lstms.at(1).size)));

        const auto& ll = m_config.general.modules->linear;
        linear = register_module("linear", Linear(ll.in_size, ll.out_size));
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        // INPUT sigs: NCT & seqs: NTC
        // spdlog::info("{} {}", utils::print_size(sigs, "sigs"), utils::print_size(seqs, "seqs"));
        // utils::dump_tensor(sigs, "sigs_in", 0);
        // utils::dump_tensor(seqs, "seqs_in", 0);

        {
            utils::ScopedProfileRange spr("sig convs", 2);
            sigs = sig_conv1(sigs);
            // spdlog::info("{}", utils::print_size(seqs, "sigs_c1_out"));
            // utils::dump_tensor(sigs, "sig_conv1", 0);
            sigs = sig_conv2(sigs);
            // utils::dump_tensor(sigs, "sig_conv2", 0);
            // spdlog::info("{}", utils::print_size(sigs, "sigs_c2_out"));
            sigs = sig_conv3(sigs);
        }

        // utils::dump_tensor(sigs, "sigs_conv_out", 0);

        // spdlog::info("{}", utils::print_size(sigs, "sigs_out"));
        // We are supplied one hot encoded sequences as (batch, signal/stride, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal/stride) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

        // seqs: NTC -> NCT - this is free as C=1
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        {
            utils::ScopedProfileRange spr("seq convs", 2);
            seqs = seq_conv1(seqs);
            // spdlog::info("{}", utils::print_size(seqs, "seqs_c1_out"));
            // utils::dump_tensor(seqs, "seq_conv1", 0);
            seqs = seq_conv2(seqs);
        }

        // utils::dump_tensor(seqs, "seqs_conv_out", 0);

        // spdlog::info("{}", utils::print_size(seqs, "seqs_out"));
        // z: NCT
        auto z = torch::cat({sigs, seqs}, 1);
        // utils::dump_tensor(z, "merge_in", 0);
        // spdlog::info("{}", utils::print_size(z, "cat"));
        {
            utils::ScopedProfileRange spr("merge convs", 2);
            // z: NCT -> TNC
            z = merge_conv1(z).permute({2, 0, 1});
        }
        // utils::dump_tensor(z, "merge_conv_out", 0);
        // spdlog::info("{}", utils::print_size(z, "merged"));
        {
            utils::ScopedProfileRange spr("lstm1", 2);
            auto [z1, h1] = lstm1(z);
            z = z1.flip(0);
        }
        // utils::dump_tensor(z, "lstm1_out", 0);
        {
            utils::ScopedProfileRange spr("lstm2", 2);
            auto [z2, h2] = lstm2(z);
            z = z2;  // .flip(0) removed
        }
        // utils::dump_tensor(z, "lstm2_out", 0);

        utils::ScopedProfileRange spr("linear", 2);
        // T'NC -> NTC
        z = linear(z).flip(0).permute({1, 0, 2}).softmax(2).flatten(1);
        // utils::dump_tensor(z, "out", 0);
        // spdlog::info("{}", utils::print_size(z, "out"));
        return z;
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights([[maybe_unused]] const std::filesystem::path& dir) {
        if (utils::get_dev_opt<int>("modbase_v3_random", 0) == 0) {
            return utils::load_tensors(dir, weight_tensors);
        }

        const auto& modules = m_config.general.modules;
        const auto& seq = modules->sequence_convs;
        const auto& sig = modules->signal_convs;
        const auto& merge = modules->merge_conv;
        const auto& rnn = modules->lstms;
        const auto& ll = modules->linear;

        std::vector<at::IntArrayRef> sizes = {
                {sig[0].size, sig[0].insize, sig[0].winlen},  // "sig_conv.conv1.weight.tensor"
                {sig[0].size},                                // "sig_conv.conv1.bias.tensor"
                {sig[1].size, sig[1].insize, sig[1].winlen},  // "sig_conv.conv2.weight.tensor"
                {sig[1].size},                                // "sig_conv.conv2.bias.tensor"
                {sig[2].size, sig[2].insize, sig[2].winlen},  // "sig_conv.conv3.weight.tensor"
                {sig[2].size},                                // "sig_conv.conv3.bias.tensor"
                {seq[0].size, seq[0].insize, seq[0].winlen},  // "seq_conv.conv1.weight.tensor"
                {seq[0].size},                                // "seq_conv.conv1.bias.tensor"
                {seq[1].size, seq[1].insize, seq[1].winlen},  // "seq_conv.conv2.weight.tensor"
                {seq[1].size},                                // "seq_conv.conv2.bias.tensor"
                {merge.size, merge.insize, merge.winlen},     // "merge_conv.conv1.weight.tensor"
                {merge.size},                                 // "merge_conv.conv1.bias.tensor"
                {4 * rnn[0].size, rnn[0].size},               // "rnns.rnn1.weight_ih_l0.tensor"
                {4 * rnn[0].size, rnn[0].size},               // "rnns.rnn1.weight_hh_l0.tensor"
                {4 * rnn[0].size},                            // "rnns.rnn1.bias_ih_l0.tensor"
                {4 * rnn[0].size},                            // "rnns.rnn1.bias_hh_l0.tensor"
                {4 * rnn[1].size, rnn[1].size},               // "rnns.rnn2.weight_ih_l0.tensor"
                {4 * rnn[1].size, rnn[1].size},               // "rnns.rnn2.weight_hh_l0.tensor"
                {4 * rnn[1].size},                            // "rnns.rnn2.bias_ih_l0.tensor"
                {4 * rnn[1].size},                            // "rnns.rnn2.bias_hh_l0.tensor"
                {ll.out_size, ll.in_size},                    // "linear.linear.weight.tensor"
                {ll.out_size},                                // "linear.linear.bias.tensor"
        };

        const auto opts = at::TensorOptions().dtype(torch::kF16);
        auto weights = std::vector<at::Tensor>();
        for (const auto& szs : sizes) {
            weights.push_back(torch::randn(szs, opts));
        }

        return weights;
    }

    static const std::vector<std::string> weight_tensors;

    const config::ModBaseModelConfig m_config;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
};

const std::vector<std::string> ModBaseConvLSTMV3ModelImpl::weight_tensors{
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

TORCH_MODULE(ModBaseConvModel);
TORCH_MODULE(ModBaseConvLSTMModel);
TORCH_MODULE(ModBaseConvLSTMV3Model);

}  // namespace nn

dorado::utils::ModuleWrapper load_modbase_model(const config::ModBaseModelConfig& config,
                                                const at::TensorOptions& options) {
    at::InferenceMode guard;
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif
    const auto params = config.general;
    switch (params.model_type) {
    case config::ModelType::CONV_LSTM_V1: {
        auto model = nn::ModBaseConvLSTMModel(params.size, params.kmer_len, params.num_out, false,
                                              params.stride);
        return populate_model(std::move(model), config.model_path, options);
    }
    case config::ModelType::CONV_LSTM_V2: {
        auto model = nn::ModBaseConvLSTMModel(params.size, params.kmer_len, params.num_out, true,
                                              params.stride);
        return populate_model(std::move(model), config.model_path, options);
    }
    case config::ModelType::CONV_LSTM_V3: {
        auto model = nn::ModBaseConvLSTMV3Model(config);
        return populate_model(std::move(model), config.model_path, options);
    }
    case config::ModelType::CONV_V1: {
        auto model = nn::ModBaseConvModel(params.size, params.kmer_len, params.num_out);
        return populate_model(std::move(model), config.model_path, options);
    }
    default:
        throw std::runtime_error("Unknown modbase model type in config file.");
    }
}

std::vector<float> load_kmer_refinement_levels(const config::ModBaseModelConfig& config) {
    std::vector<float> levels;
    if (!config.refine.do_rough_rescale) {
        return levels;
    }

    std::vector<at::Tensor> tensors =
            utils::load_tensors(config.model_path, {"refine_kmer_levels.tensor"});
    if (tensors.empty()) {
        throw std::runtime_error("Failed to load modbase refinement tensors.");
    }
    auto& t = tensors.front();
    t.contiguous();
    levels.reserve(t.numel());
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(), std::back_inserter(levels));
    return levels;
}

}  // namespace dorado::modbase
