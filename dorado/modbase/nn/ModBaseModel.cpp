#include "ModBaseModel.h"

#include "config/ModBaseModelConfig.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"
#include "torch_utils/tensor_utils.h"

#include <ATen/core/TensorBody.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
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
        activation = register_module("activation", SiLU());
    }

    at::Tensor forward(const at::Tensor& x) {
        utils::ScopedProfileRange spr(name.c_str(), 3);
        return activation(conv(x));
    }

    const std::string name;
    Conv1d conv{nullptr};
    SiLU activation{nullptr};
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

TORCH_MODULE(ModBaseConvModel);
TORCH_MODULE(ModBaseConvLSTMModel);

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
