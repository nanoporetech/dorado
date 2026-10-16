#include "Model.h"

#include "CRFModel.h"
#include "TxModel.h"
#include "config/BasecallModelConfig.h"
#include "torch_utils/tensor_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

namespace dorado::basecall::model {

namespace {

std::vector<at::Tensor> load_lstm_model_weights(const config::BasecallModelConfig& config) {
    if (!config.is_lstm_model() && !config.is_flstm_model()) {
        throw std::runtime_error("load_lstm_model_weights expected a lstm model config from: '" +
                                 config.model_path.string() + "'");
    }
    const bool decomposition = config.out_features.has_value();
    const bool linear_layer_bias = config.bias;

    const std::vector<std::string> conv_names{".conv.weight.tensor", ".conv.bias.tensor"};
    std::vector<std::string> lstm_names;
    if (config.is_flstm_model()) {
        lstm_names = {
                ".rnn.dn_weight_ih.tensor", ".rnn.dn_weight_hh.tensor", ".rnn.up_weight_ih.tensor",
                ".rnn.up_weight_hh.tensor", ".rnn.up_bias_ih.tensor",   ".rnn.up_bias_hh.tensor",
        };
    } else {
        lstm_names = {
                ".rnn.weight_ih_l0.tensor",
                ".rnn.weight_hh_l0.tensor",
                ".rnn.bias_ih_l0.tensor",
                ".rnn.bias_hh_l0.tensor",
        };
    }

    const size_t num_conv_weights = config.convs.size() * conv_names.size();
    const size_t num_rnn_weights = config.lstm_layers * lstm_names.size();
    const size_t num_tensors = num_conv_weights + num_rnn_weights + 1 + size_t(linear_layer_bias) +
                               size_t(decomposition);

    std::vector<std::string> tensors{num_tensors};

    for (size_t conv = 0; conv < config.convs.size(); ++conv) {
        for (size_t name_idx = 0; name_idx < conv_names.size(); ++name_idx) {
            const size_t idx = conv * conv_names.size() + name_idx;
            tensors.at(idx) = std::to_string(conv) + conv_names.at(name_idx);
        }
    }

    size_t offset = num_conv_weights;
    for (int lstm = 0; lstm < config.lstm_layers; ++lstm) {
        for (size_t name_idx = 0; name_idx < lstm_names.size(); ++name_idx) {
            const size_t idx = offset + lstm * lstm_names.size() + name_idx;
            const size_t layer = config.convs.size() + lstm + 1;  // skip fused layer
            tensors.at(idx) = std::to_string(layer) + lstm_names.at(name_idx);
        }
    }

    const size_t layer = config.convs.size() + config.lstm_layers + 1;
    offset += num_rnn_weights;
    tensors.at(offset) = std::to_string(layer) + ".linear.weight.tensor";
    if (linear_layer_bias) {
        tensors.at(++offset) = std::to_string(layer) + ".linear.bias.tensor";
    }
    if (decomposition) {
        tensors.at(++offset) = std::to_string(layer + 1) + ".linear.weight.tensor";
    }

    return utils::load_tensors(config.model_path, tensors);
}

std::vector<at::Tensor> load_tx_model_weights(const config::BasecallModelConfig& config) {
    if (!config.is_tx_model()) {
        throw std::runtime_error(
                "load_tx_model_weights expected a transformer model config from: '" +
                config.model_path.string() + "'");
    }

    const std::string conv_prefix = "conv.";
    const std::vector<std::string> conv_names{".conv.weight.tensor", ".conv.bias.tensor"};
    const std::string encoder_prefix = "transformer_encoder.";
    const std::vector<std::string> encoder_names{".self_attn.Wqkv.weight.tensor",
                                                 ".self_attn.out_proj.weight.tensor",
                                                 ".self_attn.out_proj.bias.tensor",
                                                 ".ff.fc1.weight.tensor",
                                                 ".ff.fc2.weight.tensor",
                                                 ".norm1.weight.tensor",
                                                 ".norm2.weight.tensor"};
    const std::vector<std::string> remaining_names{"upsample.linear.weight.tensor",
                                                   "upsample.linear.bias.tensor",
                                                   "crf.linear.weight.tensor"};

    const size_t num_conv_weights = config.convs.size() * conv_names.size();
    const size_t num_encoder_weights = config.tx->tx.depth * encoder_names.size();
    const size_t num_tensors = num_conv_weights + num_encoder_weights + remaining_names.size();
    std::vector<std::string> tensors{num_tensors};

    for (size_t conv = 0; conv < config.convs.size(); ++conv) {
        for (size_t name_idx = 0; name_idx < conv_names.size(); ++name_idx) {
            const size_t idx = conv * conv_names.size() + name_idx;
            tensors.at(idx) = conv_prefix + std::to_string(conv) + conv_names.at(name_idx);
        }
    }

    size_t offset = num_conv_weights;
    for (int encoder = 0; encoder < config.tx->tx.depth; ++encoder) {
        for (size_t name_idx = 0; name_idx < encoder_names.size(); ++name_idx) {
            const size_t idx = offset + encoder * encoder_names.size() + name_idx;
            tensors.at(idx) = encoder_prefix + std::to_string(encoder) + encoder_names.at(name_idx);
        }
    }

    offset += num_encoder_weights;
    for (size_t name_idx = 0; name_idx < remaining_names.size(); ++name_idx) {
        tensors.at(offset + name_idx) = remaining_names.at(name_idx);
    }

    return utils::load_tensors(config.model_path, tensors);
}

template <typename ModelType>
torch::nn::ModuleHolder<torch::nn::AnyModule> load_torch_model(
        const config::BasecallModelConfig& config,
        const at::TensorOptions& options) {
    auto model = ModelType(config, options);
    model->load_state_dict(load_model_weights(config));
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();
    return torch::nn::ModuleHolder<torch::nn::AnyModule>(std::move(model));
}

// Prevent maybe-uninitialized errors in ont_core_cpp sanitizer build stemming from device_guard.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
torch::nn::ModuleHolder<torch::nn::AnyModule> load_module(const config::BasecallModelConfig& config,
                                                          const at::TensorOptions& options) {
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif
    if (config.is_tx_model()) {
        return load_torch_model<TxModel>(config, options);
    }
    return load_torch_model<CRFModel>(config, options);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

}  // namespace

Model::Model(torch::nn::ModuleHolder<torch::nn::AnyModule> module) : m_module(std::move(module)) {}

ModelOutput Model::forward(const at::Tensor& input,
                           nn::AuxiliaryData* aux,
                           bool collect_signal_embeddings) {
    ModelOutput output;
    output.scores = collect_signal_embeddings
                            ? m_module->forward(input, aux, &output.signal_embeddings)
                            : m_module->forward(input, aux);
    return output;
}

std::unique_ptr<Model> load_model(const config::BasecallModelConfig& model_config,
                                  const at::TensorOptions& options) {
    return std::make_unique<Model>(load_module(model_config, options));
}

std::vector<at::Tensor> load_model_weights(const config::BasecallModelConfig& model_config) {
    return model_config.is_tx_model() ? load_tx_model_weights(model_config)
                                      : load_lstm_model_weights(model_config);
}

}  // namespace dorado::basecall::model
