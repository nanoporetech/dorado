#include "crf_utils.h"

#include "config/BasecallModelConfig.h"
#include "nn/CRFModel.h"
#include "nn/TxModel.h"
#include "torch_utils/tensor_utils.h"
#include "utils/memory_utils.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dorado::basecall {

namespace {

using namespace torch::nn;
using namespace config;

std::vector<at::Tensor> load_lstm_model_weights(const std::filesystem::path &dir,
                                                bool decomposition,
                                                bool linear_layer_bias) {
    auto tensors = std::vector<std::string>{

            "0.conv.weight.tensor",      "0.conv.bias.tensor",

            "1.conv.weight.tensor",      "1.conv.bias.tensor",

            "2.conv.weight.tensor",      "2.conv.bias.tensor",

            "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
            "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

            "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
            "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

            "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
            "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

            "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
            "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

            "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
            "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

            "9.linear.weight.tensor"};

    if (linear_layer_bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return utils::load_tensors(dir, tensors);
}

std::vector<torch::Tensor> load_tx_model_weights(const BasecallModelConfig &cfg) {
    if (!cfg.is_tx_model()) {
        throw std::runtime_error(
                "load_tx_model_weights expected a transformer model config from: '" +
                cfg.model_path.string() + "'");
    }

    const std::string conv_prefix = "conv.";
    const std::vector<std::string> conv_names{".conv.weight.tensor", ".conv.bias.tensor"};

    const std::string enc_prefix = "transformer_encoder.";
    const std::vector<std::string> enc_names{".self_attn.Wqkv.weight.tensor",
                                             ".self_attn.out_proj.weight.tensor",
                                             ".self_attn.out_proj.bias.tensor",
                                             ".ff.fc1.weight.tensor",
                                             ".ff.fc2.weight.tensor",
                                             ".norm1.weight.tensor",
                                             ".norm2.weight.tensor"};

    const std::vector<std::string> remaining_names{"upsample.linear.weight.tensor",
                                                   "upsample.linear.bias.tensor",
                                                   "crf.linear.weight.tensor"};

    const size_t num_conv_weights = cfg.convs.size() * conv_names.size();
    const size_t num_tx_weights = cfg.tx->tx.depth * enc_names.size();
    const size_t num_tensors = num_conv_weights + num_tx_weights + remaining_names.size();

    auto tensors = std::vector<std::string>{num_tensors};

    // Add convolutions
    for (size_t cv = 0; cv < cfg.convs.size(); cv++) {
        for (size_t n = 0; n < conv_names.size(); n++) {
            const size_t idx = cv * conv_names.size() + n;
            const std::string name = conv_prefix + std::to_string(cv) + conv_names.at(n);
            tensors.at(idx) = name;
        }
    }

    // Add encoders
    size_t offset = num_conv_weights;
    for (int enc = 0; enc < cfg.tx->tx.depth; enc++) {
        for (size_t n = 0; n < enc_names.size(); n++) {
            const size_t idx = offset + enc * enc_names.size() + n;
            const std::string name = enc_prefix + std::to_string(enc) + enc_names.at(n);
            tensors.at(idx) = name;
        }
    }

    // Add upsample and linear
    offset += num_tx_weights;
    for (size_t i = 0; i < remaining_names.size(); i++) {
        const size_t idx = offset + i;
        const std::string &name = remaining_names.at(i);
        tensors.at(idx) = name;
    }

    return utils::load_tensors(cfg.model_path, tensors);
}

}  // namespace

std::vector<at::Tensor> load_crf_model_weights(const BasecallModelConfig &model_config) {
    if (model_config.is_tx_model()) {
        return load_tx_model_weights(model_config);
    }
    return load_lstm_model_weights(model_config.model_path, model_config.out_features.has_value(),
                                   model_config.bias);
}

namespace {

ModuleHolder<AnyModule> load_lstm_model(const BasecallModelConfig &model_config,
                                        const at::TensorOptions &options) {
    auto model = nn::CRFModel(model_config);
    auto state_dict = load_crf_model_weights(model_config);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}

ModuleHolder<AnyModule> load_tx_model(const BasecallModelConfig &model_config,
                                      const at::TensorOptions &options) {
    auto model = nn::TxModel(model_config, options);
    auto state_dict = load_crf_model_weights(model_config);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}

}  // namespace

ModuleHolder<AnyModule> load_crf_model(const BasecallModelConfig &model_config,
                                       const torch::TensorOptions &options) {
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif
    if (model_config.is_tx_model()) {
        return load_tx_model(model_config, options);
    }
    return load_lstm_model(model_config, options);
}

size_t auto_calculate_num_runners(const BasecallModelConfig &model_config, float memory_fraction) {
    auto model_name = std::filesystem::canonical(model_config.model_path).filename().string();

    // very hand-wavy determination
    // these numbers were determined empirically by running 1, 2, 4 and 8 runners for each model
    auto required_ram_per_runner_GB = 0.f;
    if (model_name.find("_fast@v") != std::string::npos) {
        required_ram_per_runner_GB = 1.5;
    } else if (model_name.find("_hac@v") != std::string::npos) {
        required_ram_per_runner_GB = 4.5;
    } else if (model_name.find("_sup@v") != std::string::npos) {
        required_ram_per_runner_GB = 12.5;
    } else {
        return 1;
    }

    // Should have set batch_size to non-zero value if device == cpu
    assert(model_config.basecaller.batch_size() > 0);
    // numbers were determined with a batch_size of 128, assume this just scales
    required_ram_per_runner_GB *= model_config.basecaller.batch_size() / 128.f;

    auto free_ram_GB = utils::available_host_memory_GB() * memory_fraction;
    auto num_runners = static_cast<size_t>(free_ram_GB / required_ram_per_runner_GB);
    return std::clamp(num_runners, size_t(1), std::size_t(std::thread::hardware_concurrency()));
}

}  // namespace dorado::basecall
