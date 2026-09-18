#include "ModBaseModel.h"

#include "config/ModBaseModelConfig.h"
#include "nn/LinearUpsample.h"
#include "nn/TxModules.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/overloaded.h"

#include <ATen/core/TensorBody.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <optional>
#include <string_view>

#if DORADO_CUDA_BUILD
#include "nn/ConvStack.h"
#include "nn/KoiUtils.h"
#include "nn/LSTMStack.h"

#include <c10/cuda/CUDAGuard.h>
#endif

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

    auto module = AnyModule(std::forward<Model>(model));
    auto holder = ModuleHolder<AnyModule>(std::move(module));
    return holder;
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

std::string indexed_layer_name(const std::string_view name,
                               const size_t index,
                               const size_t layer_count) {
    if (layer_count == 1) {
        return std::string(name);
    }
    return std::string(name) + std::to_string(index + 1);
}

std::vector<std::string> load_latent_space_v1_common_weight_names(
        const dorado::config::modbase::LatentSpaceModulesParams& modules) {
    std::vector<std::string> tensors;
    tensors.reserve((modules.sequence_convs.size() * 2) + (modules.projection.size() * 2) +
                    (modules.transformer_encoder.size() * 7));

    for (size_t i = 0; i < modules.sequence_convs.size(); ++i) {
        const auto prefix = "seq_conv" + std::to_string(i + 1);
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    for (size_t i = 0; i < modules.projection.size(); ++i) {
        const auto prefix = indexed_layer_name("proj", i, modules.projection.size());
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    for (size_t i = 0; i < modules.transformer_encoder.size(); ++i) {
        const auto prefix = "transformer" + std::to_string(i + 1);
        tensors.push_back(prefix + ".Wqkv.weight.tensor");
        tensors.push_back(prefix + ".out_proj.weight.tensor");
        tensors.push_back(prefix + ".out_proj.bias.tensor");
        tensors.push_back(prefix + ".fc1.weight.tensor");
        tensors.push_back(prefix + ".fc2.weight.tensor");
        tensors.push_back(prefix + ".norm1.weight.tensor");
        tensors.push_back(prefix + ".norm2.weight.tensor");
    }
    return tensors;
}

std::vector<std::string> load_latent_space_v1_final_weight_names(
        const dorado::config::modbase::LatentSpaceModulesParams& modules) {
    std::vector<std::string> tensors;
    tensors.reserve(modules.classifier.size() * 2);
    for (size_t i = 0; i < modules.classifier.size(); ++i) {
        const auto prefix = indexed_layer_name("fc", i, modules.classifier.size());
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    return tensors;
}

bool conv_params_equal(const dorado::config::ConvParams& lhs,
                       const dorado::config::ConvParams& rhs) {
    return std::tie(lhs.insize, lhs.size, lhs.winlen, lhs.stride, lhs.activation, lhs.inner_dim) ==
           std::tie(rhs.insize, rhs.size, rhs.winlen, rhs.stride, rhs.activation, rhs.inner_dim);
}

bool linear_params_equal(const dorado::config::modbase::LinearParams& lhs,
                         const dorado::config::modbase::LinearParams& rhs) {
    return std::tie(lhs.in_size, lhs.out_size) == std::tie(rhs.in_size, rhs.out_size);
}

bool transformer_params_equal(const dorado::config::TransformerEncoderLayerParams& lhs,
                              const dorado::config::TransformerEncoderLayerParams& rhs) {
    return std::tie(lhs.d_model, lhs.nhead, lhs.dim_feedforward, lhs.attn_window,
                    lhs.deepnorm_alpha, lhs.theta, lhs.max_seq_len) ==
           std::tie(rhs.d_model, rhs.nhead, rhs.dim_feedforward, rhs.attn_window,
                    rhs.deepnorm_alpha, rhs.theta, rhs.max_seq_len);
}

template <typename T, typename Compare>
bool vectors_equal(const std::vector<T>& lhs, const std::vector<T>& rhs, Compare compare) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(), compare);
}

bool common_configs_equal(const dorado::config::modbase::LatentSpaceV1ModelConfig& lhs,
                          const dorado::config::modbase::LatentSpaceV1ModelConfig& rhs) {
    // num_out and the classifier are intentionally excluded: they belong to the final model.
    return (std::tie(lhs.general.size, lhs.general.kmer_len, lhs.general.stride,
                     lhs.chunked.chunk_size, lhs.sequence_stride) ==
            std::tie(rhs.general.size, rhs.general.kmer_len, rhs.general.stride,
                     rhs.chunked.chunk_size, rhs.sequence_stride)) &&
           vectors_equal(lhs.modules.sequence_convs, rhs.modules.sequence_convs,
                         conv_params_equal) &&
           vectors_equal(lhs.modules.projection, rhs.modules.projection, linear_params_equal) &&
           vectors_equal(lhs.modules.transformer_encoder, rhs.modules.transformer_encoder,
                         transformer_params_equal);
}

bool model_parameters_equal(const torch::nn::Module& lhs, const torch::nn::Module& rhs) {
    const auto lhs_params = lhs.parameters();
    const auto rhs_params = rhs.parameters();
    if (lhs_params.size() != rhs_params.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs_params.size(); ++i) {
        const auto& lhs_param = lhs_params.at(i);
        const auto& rhs_param = rhs_params.at(i);
        if (lhs_param.sizes() != rhs_param.sizes() ||
            lhs_param.scalar_type() != rhs_param.scalar_type()) {
            return false;
        }
        const auto lhs_cpu = lhs_param.device().is_cpu() ? lhs_param : lhs_param.cpu();
        const auto rhs_cpu = rhs_param.device().is_cpu() ? rhs_param : rhs_param.cpu();
        if (!lhs_cpu.equal(rhs_cpu)) {
            return false;
        }
    }
    return true;
}

}  // namespace

#if DORADO_CUDA_BUILD
namespace dorado {
namespace {
std::pair<int64_t, int64_t> get_chunked_sequence_input_TC(
        const config::modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size / (config.general.stride / config.sequence_stride),
            config.data.context.kmer_len * 4};
}

std::pair<int64_t, int64_t> get_chunked_signal_input_TC(
        const config::modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size, 1};
}
}  // namespace
}  // namespace dorado
#endif

namespace dorado::modbase {

namespace model {

struct ModsConvImpl : Module {
    ModsConvImpl(int size, int outsize, int k, int stride, int padding)
            : name("conv_act_" + std::to_string(size)) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(padding)));
        activation = config::Activation::SWISH;
    }

    explicit ModsConvImpl(const config::ConvParams& params)
            : name("conv_act_" + std::to_string(params.size)) {
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
    explicit ModBaseConvModelImpl(const config::modbase::ConvOnlyModelConfig& config) {
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
        return utils::load_tensors(dir, weight_tensors());
    }

    static std::vector<std::string> weight_tensors();

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

struct ModBaseConvLSTMModelImpl : Module {
    explicit ModBaseConvLSTMModelImpl(const config::modbase::ConvLSTMModelConfig& config)
            : m_is_conv_lstm_v2(false) {
        initialise(config.general);
    }

    explicit ModBaseConvLSTMModelImpl(const config::modbase::ConvLSTMV2ModelConfig& config)
            : m_is_conv_lstm_v2(true) {
        initialise(config.general);
    }

    void initialise(const config::modbase::CommonModelParams& params) {
        const int signal_stride = params.stride;
        // conv_lstm_v2 models are padded to ensure the output shape is nicely indexable by the stride
        sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv3 = register_module("sig_conv3", ModsConv(16, params.size, 9, signal_stride,
                                                          m_is_conv_lstm_v2 ? 4 : 0));

        seq_conv1 = register_module(
                "seq_conv1", ModsConv(params.kmer_len * 4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        seq_conv2 = register_module("seq_conv2", ModsConv(16, params.size, 13, params.stride,
                                                          m_is_conv_lstm_v2 ? 6 : 0));

        merge_conv1 = register_module("merge_conv1", ModsConv(params.size * 2, params.size, 5, 1,
                                                              m_is_conv_lstm_v2 ? 2 : 0));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(params.size, params.size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(params.size, params.size)));

        linear = register_module("linear", Linear(params.size, params.num_out));

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
        return utils::load_tensors(dir, load_modbase_conv_lstm_weights(false));
    }

    const bool m_is_conv_lstm_v2{false};

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

struct ModBaseConvLSTMV3ModelImpl : Module {
    explicit ModBaseConvLSTMV3ModelImpl(const config::modbase::ConvLSTMV3ModelConfig& config)
            : m_config(config) {
        const auto& sig_cvs = m_config.modules.signal_convs;
        const auto& seq_cvs = m_config.modules.sequence_convs;
        const auto& merge_cv = m_config.modules.merge_conv;
        const auto& lstms = m_config.modules.lstms;
        const auto& ll = m_config.modules.linear;
        const auto& lu = m_config.modules.upsample;

        if (sig_cvs.size() != 3) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
        }
        if (seq_cvs.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
        }
        if (lstms.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
        }

        sig_conv1 = register_module("sig_conv1", ModsConv(sig_cvs.at(0)));
        sig_conv2 = register_module("sig_conv2", ModsConv(sig_cvs.at(1)));
        sig_conv3 = register_module("sig_conv3", ModsConv(sig_cvs.at(2)));

        seq_conv1 = register_module("seq_conv1", ModsConv(seq_cvs.at(0)));
        seq_conv2 = register_module("seq_conv2", ModsConv(seq_cvs.at(1)));

        merge_conv1 = register_module("merge_conv1", ModsConv(merge_cv));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(lstms.at(0).size, lstms.at(0).size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(lstms.at(1).size, lstms.at(1).size)));

        linear = register_module("linear", Linear(ll.in_size, ll.out_size));

        if (lu.has_value()) {
            upsample = register_module("upsample", nn::LinearUpsample(lu.value()));
        }
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        // INPUT sigs: NCT & seqs: NTC
        {
            utils::ScopedProfileRange spr("sig convs", 2);
            sigs = sig_conv1(sigs);
            sigs = sig_conv2(sigs);
            sigs = sig_conv3(sigs);
        }

        // We are supplied one hot encoded sequences as (batch, signal/stride, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal/stride) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

        // seqs: NTC -> NCT
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        {
            utils::ScopedProfileRange spr("seq convs", 2);
            seqs = seq_conv1(seqs);
            seqs = seq_conv2(seqs);
        }

        // z: NCT -> N(2C)T
        auto z = torch::cat({sigs, seqs}, 1);
        {
            utils::ScopedProfileRange spr("merge convs", 2);
            z = merge_conv1(z).permute({2, 0, 1});  // NCT -> TNC
        }
        {
            utils::ScopedProfileRange spr("lstm1", 2);
            z = std::get<0>(lstm1(z)).flip(0);  // TNC -> T'NC
        }
        {
            utils::ScopedProfileRange spr("lstm2", 2);
            z = std::get<0>(lstm2(z));  // T'NC
        }
        {
            utils::ScopedProfileRange spr("linear", 2);
            z = linear(z).flip(0).permute({1, 0, 2});  // T'NC -> NTC
        }
        if (m_config.modules.upsample.has_value()) {
            utils::ScopedProfileRange spr2("upsample", 2);
            z = upsample->forward(z);  // NTC -> N(sf*T)C
        }
        utils::ScopedProfileRange spr2("softmax", 2);
        // NTC -> N(TC)
        return z.softmax(2).flatten(1);
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(
                dir, load_modbase_conv_lstm_weights(m_config.modules.upsample.has_value()));
    }

    const config::modbase::ConvLSTMV3ModelConfig m_config;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    nn::LinearUpsample upsample{nullptr};
};

#if DORADO_CUDA_BUILD

struct ModBaseConvLSTMV3CUDAModelImpl : Module {
    ModBaseConvLSTMV3CUDAModelImpl(const config::modbase::ConvLSTMV3ModelConfig& config,
                                   const int batch_size_)
            : m_config(config), batch_size(batch_size_) {
        const auto& sig_cvs = m_config.modules.signal_convs;
        const auto& seq_cvs = m_config.modules.sequence_convs;
        const auto& merge_cv = m_config.modules.merge_conv;
        const auto& lstms = m_config.modules.lstms;
        const auto& ll = m_config.modules.linear;
        const auto& lu = m_config.modules.upsample;

        if (sig_cvs.size() != 3) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
        }
        if (seq_cvs.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
        }
        if (lstms.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
        }

        sig_convs = register_module("sig", nn::ConvStack(sig_cvs));
        seq_convs = register_module("seq", nn::ConvStack(seq_cvs));
        merge_conv = register_module("merge", nn::ConvStack(std::vector{merge_cv}));
        lstm = register_module("lstm", nn::LSTMStack(lstms.size(), lstms.at(0).size, false));
        linear = register_module("linear", Linear(ll.in_size, ll.out_size));
        if (lu.has_value()) {
            upsample = register_module("upsample", nn::LinearUpsample(lu.value()));
        }
        {
            // Reserve and allocate working memory for the sequence convs
            const auto [T, C] = get_chunked_sequence_input_TC(m_config);
            wm_seq.next_TC(T, C, nn::TensorLayout::NTC);
            seq_convs->reserve_working_memory(wm_seq, nullptr, nn::TensorLayout::NTC);
        }
        {
            // Reserve and allocate working memory for everything else
            const auto [T, C] = get_chunked_signal_input_TC(m_config);
            wm.next_TC(T, C, nn::TensorLayout::NTC);
            sig_convs->reserve_working_memory(wm, nullptr, nn::TensorLayout::NTC);

            // Buffer for concatenating conv outputs before merge_conv
            const int padding = merge_cv.winlen / 2;
            const int64_t T_merge_conv = (T / m_config.modules.signal_stride()) + (2 * padding);
            wm.next_TC(T_merge_conv, merge_cv.size, nn::TensorLayout::NTC);

            std::optional<nn::TensorLayout> force_dtype = std::nullopt;
            if (nn::koi_can_use_cutlass()) {
                force_dtype = utils::get_dev_opt("modbase_i8", true)
                                      ? nn::TensorLayout::CUTLASS_TNC_I8
                                      : nn::TensorLayout::CUTLASS_TNC_F16;
                spdlog::trace("Set modbase merge_conv CUTLASS dtype: {}",
                              to_string(force_dtype.value()));
            }

            merge_conv->reserve_working_memory(wm, nullptr, force_dtype);
            lstm->reserve_working_memory(wm);
        }
    }

    at::Tensor forward(const at::Tensor& sigs_N1T, const at::Tensor& seqs_NTC) {
        utils::ScopedProfileRange spr("mbv3 koi", 1);

        c10::cuda::CUDAGuard device_guard(sigs_N1T.device());

        {
            utils::ScopedProfileRange spr2("seq_copy", 2);
            const auto [Tin_seq, Cin_seq] = get_chunked_sequence_input_TC(m_config);
            if (!wm_seq.backing_tensor.defined()) {
                wm_seq.allocate_backing_tensor(seqs_NTC.device());
            }
            auto wm_seq_in = wm_seq.next_TC(Tin_seq, Cin_seq, nn::TensorLayout::NTC);
            wm_seq_in.index({Slice()}) = seqs_NTC;
        }
        {
            utils::ScopedProfileRange spr2("sig_copy", 2);
            const auto [Tin_sig, Cin_sig] = get_chunked_signal_input_TC(m_config);
            if (!wm.backing_tensor.defined()) {
                wm.allocate_backing_tensor(sigs_N1T.device());
            }
            auto wm_in = wm.next_TC(Tin_sig, Cin_sig, nn::TensorLayout::NTC);
            // This transpose is free because C is 1
            wm_in.index({Slice()}) = sigs_N1T.transpose(1, 2);
        }
        {
            utils::ScopedProfileRange spr2("seq_convs", 2);
            seq_convs->run_koi(wm_seq, nullptr);
        }
        {
            utils::ScopedProfileRange spr2("sig_convs", 2);
            sig_convs->run_koi(wm, nullptr);
        }

        if (wm.layout != wm_seq.layout) {
            throw std::runtime_error(
                    "Signal and sequence convolution stacks must have the same output "
                    "layout");
        }
        {
            // TODO: This can be removed with changes to the sequence and signal convolutions
            // Copy the output of seq_convs stored in wm_seq into wm for the merge_conv input buffer.
            utils::ScopedProfileRange spr2("merge_prep", 2);

            const auto& merge_cv = m_config.modules.merge_conv;
            auto seq_out = wm_seq.current;
            auto sig_out = wm.current;

            // If the merge conv is using CUTLASS - need to add padding
            const auto mcv_layout = merge_conv->layers.back().output_layout;
            const bool using_cutlass = mcv_layout == nn::TensorLayout::CUTLASS_TNC_F16 ||
                                       mcv_layout == nn::TensorLayout::CUTLASS_TNC_I8;

            const int padding = using_cutlass ? merge_cv.winlen / 2 : 0;
            const int T = seq_out.size(1);
            const int Tp = T + (2 * padding);
            const size_t C_sum = seq_out.size(2) + sig_out.size(2);

            // Performing torch::cat({sig, seq}, 1) but with optional padding
            auto merge_conv_in =
                    wm.next_TC(Tp, C_sum, nn::TensorLayout::NTC).view({batch_size, Tp, 2, -1});
            merge_conv_in.index({Slice(), Slice(padding, T + padding), 0}) = sig_out;
            merge_conv_in.index({Slice(), Slice(padding, T + padding), 1}) = seq_out;
        }
        {
            utils::ScopedProfileRange spr2("merge_conv", 2);
            wm.is_input_to_rev_lstm = false;
            merge_conv->run_koi(wm, nullptr);
        }
        {
            utils::ScopedProfileRange spr2("lstms", 2);
            lstm->run_koi(wm, nullptr);
        }

        at::Tensor out;
        {
            utils::ScopedProfileRange spr2("linear", 2);
            out = linear->forward(wm.layout == nn::TensorLayout::CUTLASS_TNC_I8
                                          ? wm.get_current_NTC_view().to(torch::kF16) / 127.0f
                                          : wm.get_current_NTC_view());
        }
        if (m_config.modules.upsample.has_value()) {
            utils::ScopedProfileRange spr2("upsample", 2);
            out = upsample->forward(out);  // NTC -> N(sf*T)C
        }

        utils::ScopedProfileRange spr2("softmax", 2);
        return out.softmax(2).flatten(1);  // NTC -> N(TC)
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(
                dir, load_modbase_conv_lstm_weights(m_config.modules.upsample.has_value()));
    }

    const config::modbase::ConvLSTMV3ModelConfig m_config;

    nn::ConvStack sig_convs{nullptr};
    nn::ConvStack seq_convs{nullptr};
    nn::ConvStack merge_conv{nullptr};
    nn::LSTMStack lstm{nullptr};
    Linear linear{nullptr};
    nn::LinearUpsample upsample{nullptr};

    const int batch_size;
    nn::WorkingMemory wm{batch_size}, wm_seq{batch_size};
};

#endif

/**
 * The reusable part of a latent-space modification model.
 *
 * Both inputs use NTC layout. The returned latent-space tensor also uses NTC layout and can be
 * passed to one or more final models.
 */
struct ModBaseLatentSpaceV1CommonModelImpl : Module {
    explicit ModBaseLatentSpaceV1CommonModelImpl(
            const config::modbase::LatentSpaceV1ModelConfig& config,
            const at::TensorOptions& options)
            : m_config(config) {
        sequence_encoder = register_module("sequence_encoder", Sequential());
        for (const auto& conv : m_config.modules.sequence_convs) {
            sequence_encoder->push_back(ModsConv(conv));
        }

        projection = register_module("projection", Sequential());
        for (const auto& linear : m_config.modules.projection) {
            projection->push_back(Linear(linear.in_size, linear.out_size));
        }

        common_encoder = register_module("common_encoder", Sequential());
        for (auto params : m_config.modules.transformer_encoder) {
            if (params.max_seq_len == 0) {
                params.max_seq_len =
                        static_cast<int>(m_config.chunked.chunk_size / m_config.general.stride);
            }
            common_encoder->push_back(nn::TxEncoder(params, options));
        }
    }

    at::Tensor forward(at::Tensor sequences, at::Tensor embedded_features) {
        const auto conv_dtype =
                (sequences.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;
        // NTC -> NCT -> NTC
        sequences = sequence_encoder->forward(sequences.permute({0, 2, 1}).to(conv_dtype));
        sequences = sequences.permute({0, 2, 1});
        embedded_features = projection->forward(embedded_features);
        return common_encoder->forward(sequences + embedded_features);
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) const {
        return utils::load_tensors(dir, load_latent_space_v1_common_weight_names(m_config.modules));
    }

    bool is_compatible(const ModBaseLatentSpaceV1CommonModelImpl& other) const {
        return common_configs_equal(m_config, other.m_config) &&
               model_parameters_equal(*this, other);
    }

    bool operator==(const ModBaseLatentSpaceV1CommonModelImpl& other) const {
        return is_compatible(other);
    }

    const config::modbase::LatentSpaceV1ModelConfig m_config;
    Sequential sequence_encoder{nullptr};
    Sequential projection{nullptr};
    Sequential common_encoder{nullptr};
};

/** The model-specific classifier applied to the output of a latent-space common model. */
struct ModBaseLatentSpaceV1ClassifierModelImpl : Module {
    explicit ModBaseLatentSpaceV1ClassifierModelImpl(
            const config::modbase::LatentSpaceV1ModelConfig& config) {
        classifier_head = register_module("classifier_head", Sequential());
        for (const auto& linear : m_config.modules.classifier) {
            classifier_head->push_back(Linear(linear.in_size, linear.out_size));
        }
    }

    // Input is NTC; output logits are NCT.
    at::Tensor forward(at::Tensor common_output) {
        return classifier_head->forward(common_output).permute({0, 2, 1});
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) const {
        return utils::load_tensors(dir, load_latent_space_v1_final_weight_names(m_config.modules));
    }

    const config::modbase::LatentSpaceV1ModelConfig m_config;
    Sequential classifier_head{nullptr};
};

TORCH_MODULE(ModBaseConvModel);
TORCH_MODULE(ModBaseConvLSTMModel);
TORCH_MODULE(ModBaseConvLSTMV3Model);
#if DORADO_CUDA_BUILD
TORCH_MODULE(ModBaseConvLSTMV3CUDAModel);
#endif
TORCH_MODULE(ModBaseLatentSpaceV1CommonModel);
TORCH_MODULE(ModBaseLatentSpaceV1ClassifierModel);

}  // namespace model

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const config::ModBaseModelConfig& config,
        const at::TensorOptions& options,
        [[maybe_unused]] const int batchsize) {
    at::InferenceMode guard;
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif
    return std::visit(
            utils::overloaded{
                    [&](const config::modbase::ConvOnlyModelConfig& model_config) {
                        auto model = model::ModBaseConvModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMModelConfig& model_config) {
                        auto model = model::ModBaseConvLSTMModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMV2ModelConfig& model_config) {
                        auto model = model::ModBaseConvLSTMModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMV3ModelConfig& model_config) {
#if DORADO_CUDA_BUILD
                        static bool use_torch = utils::get_dev_opt("modbase_torch", false);
                        if (options.device().is_cuda() && !use_torch) {
                            auto model = model::ModBaseConvLSTMV3CUDAModel(model_config, batchsize);
                            return populate_model(std::move(model), model_config.data.model_path,
                                                  options);
                        }
#endif
                        spdlog::trace("Loading ModBaseConvLSTMV3Model - Torch only");
                        auto model = model::ModBaseConvLSTMV3Model(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::LatentSpaceV1ModelConfig&)
                            -> torch::nn::ModuleHolder<torch::nn::AnyModule> {
                        throw std::runtime_error(
                                "Latent space modbase model inference is not implemented");
                    },
            },
            config);
}

}  // namespace dorado::modbase
