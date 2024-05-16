#include "crf_utils.h"

#include "CRFModelConfig.h"
#include "nn/CRFModel.h"
#include "nn/TxModel.h"
#include "utils/memory_utils.h"
#include "utils/tensor_utils.h"

#include <algorithm>
#include <thread>

using namespace torch::nn;

namespace dorado::basecall {
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

std::vector<torch::Tensor> load_tx_model_weights(const std::filesystem::path &dir) {
    auto tensors = std::vector<std::string>{
            // convs 0-4
            "conv.0.conv.weight.tensor",
            "conv.0.conv.bias.tensor",
            "conv.1.conv.weight.tensor",
            "conv.1.conv.bias.tensor",
            "conv.2.conv.weight.tensor",
            "conv.2.conv.bias.tensor",
            "conv.3.conv.weight.tensor",
            "conv.3.conv.bias.tensor",
            "conv.4.conv.weight.tensor",
            "conv.4.conv.bias.tensor",

            // tx encoder layer 0
            "transformer_encoder.0.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.bias.tensor",
            "transformer_encoder.0.ff.fc1.weight.tensor",
            "transformer_encoder.0.ff.fc2.weight.tensor",
            "transformer_encoder.0.norm1.weight.tensor",
            "transformer_encoder.0.norm2.weight.tensor",

            // tx encoder layer 1
            "transformer_encoder.1.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.bias.tensor",
            "transformer_encoder.1.ff.fc1.weight.tensor",
            "transformer_encoder.1.ff.fc2.weight.tensor",
            "transformer_encoder.1.norm1.weight.tensor",
            "transformer_encoder.1.norm2.weight.tensor",

            // tx encoder layer 2
            "transformer_encoder.2.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.bias.tensor",
            "transformer_encoder.2.ff.fc1.weight.tensor",
            "transformer_encoder.2.ff.fc2.weight.tensor",
            "transformer_encoder.2.norm1.weight.tensor",
            "transformer_encoder.2.norm2.weight.tensor",

            // tx encoder layer 3
            "transformer_encoder.3.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.bias.tensor",
            "transformer_encoder.3.ff.fc1.weight.tensor",
            "transformer_encoder.3.ff.fc2.weight.tensor",
            "transformer_encoder.3.norm1.weight.tensor",
            "transformer_encoder.3.norm2.weight.tensor",

            // tx encoder layer 4
            "transformer_encoder.4.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.bias.tensor",
            "transformer_encoder.4.ff.fc1.weight.tensor",
            "transformer_encoder.4.ff.fc2.weight.tensor",
            "transformer_encoder.4.norm1.weight.tensor",
            "transformer_encoder.4.norm2.weight.tensor",

            // tx encoder layer 5
            "transformer_encoder.5.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.bias.tensor",
            "transformer_encoder.5.ff.fc1.weight.tensor",
            "transformer_encoder.5.ff.fc2.weight.tensor",
            "transformer_encoder.5.norm1.weight.tensor",
            "transformer_encoder.5.norm2.weight.tensor",

            // tx encoder layer 6
            "transformer_encoder.6.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.bias.tensor",
            "transformer_encoder.6.ff.fc1.weight.tensor",
            "transformer_encoder.6.ff.fc2.weight.tensor",
            "transformer_encoder.6.norm1.weight.tensor",
            "transformer_encoder.6.norm2.weight.tensor",

            // tx encoder layer 7
            "transformer_encoder.7.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.bias.tensor",
            "transformer_encoder.7.ff.fc1.weight.tensor",
            "transformer_encoder.7.ff.fc2.weight.tensor",
            "transformer_encoder.7.norm1.weight.tensor",
            "transformer_encoder.7.norm2.weight.tensor",

            // tx encoder layer 8
            "transformer_encoder.8.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.bias.tensor",
            "transformer_encoder.8.ff.fc1.weight.tensor",
            "transformer_encoder.8.ff.fc2.weight.tensor",
            "transformer_encoder.8.norm1.weight.tensor",
            "transformer_encoder.8.norm2.weight.tensor",

            // tx encoder layer 9
            "transformer_encoder.9.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.bias.tensor",
            "transformer_encoder.9.ff.fc1.weight.tensor",
            "transformer_encoder.9.ff.fc2.weight.tensor",
            "transformer_encoder.9.norm1.weight.tensor",
            "transformer_encoder.9.norm2.weight.tensor",

            // tx encoder layer 10
            "transformer_encoder.10.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.bias.tensor",
            "transformer_encoder.10.ff.fc1.weight.tensor",
            "transformer_encoder.10.ff.fc2.weight.tensor",
            "transformer_encoder.10.norm1.weight.tensor",
            "transformer_encoder.10.norm2.weight.tensor",

            // tx encoder layer 11
            "transformer_encoder.11.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.bias.tensor",
            "transformer_encoder.11.ff.fc1.weight.tensor",
            "transformer_encoder.11.ff.fc2.weight.tensor",
            "transformer_encoder.11.norm1.weight.tensor",
            "transformer_encoder.11.norm2.weight.tensor",

            // tx encoder layer 12
            "transformer_encoder.12.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.bias.tensor",
            "transformer_encoder.12.ff.fc1.weight.tensor",
            "transformer_encoder.12.ff.fc2.weight.tensor",
            "transformer_encoder.12.norm1.weight.tensor",
            "transformer_encoder.12.norm2.weight.tensor",

            // tx encoder layer 13
            "transformer_encoder.13.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.bias.tensor",
            "transformer_encoder.13.ff.fc1.weight.tensor",
            "transformer_encoder.13.ff.fc2.weight.tensor",
            "transformer_encoder.13.norm1.weight.tensor",
            "transformer_encoder.13.norm2.weight.tensor",

            // tx encoder layer 14
            "transformer_encoder.14.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.bias.tensor",
            "transformer_encoder.14.ff.fc1.weight.tensor",
            "transformer_encoder.14.ff.fc2.weight.tensor",
            "transformer_encoder.14.norm1.weight.tensor",
            "transformer_encoder.14.norm2.weight.tensor",

            // tx encoder layer 15
            "transformer_encoder.15.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.bias.tensor",
            "transformer_encoder.15.ff.fc1.weight.tensor",
            "transformer_encoder.15.ff.fc2.weight.tensor",
            "transformer_encoder.15.norm1.weight.tensor",
            "transformer_encoder.15.norm2.weight.tensor",

            // tx encoder layer 16
            "transformer_encoder.16.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.bias.tensor",
            "transformer_encoder.16.ff.fc1.weight.tensor",
            "transformer_encoder.16.ff.fc2.weight.tensor",
            "transformer_encoder.16.norm1.weight.tensor",
            "transformer_encoder.16.norm2.weight.tensor",

            // tx encoder layer 17
            "transformer_encoder.17.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.bias.tensor",
            "transformer_encoder.17.ff.fc1.weight.tensor",
            "transformer_encoder.17.ff.fc2.weight.tensor",
            "transformer_encoder.17.norm1.weight.tensor",
            "transformer_encoder.17.norm2.weight.tensor",

            // tx decoder
            "upsample.linear.weight.tensor",
            "upsample.linear.bias.tensor",

            // linear CRF
            "crf.linear.weight.tensor",
    };

    return utils::load_tensors(dir, tensors);
}

std::vector<at::Tensor> load_crf_model_weights(const CRFModelConfig &model_config) {
    if (model_config.is_tx_model()) {
        return load_tx_model_weights(model_config.model_path);
    }
    return load_lstm_model_weights(model_config.model_path, model_config.out_features.has_value(),
                                   model_config.bias);
}

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config,
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

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config,
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

ModuleHolder<AnyModule> load_crf_model(const CRFModelConfig &model_config,
                                       const torch::TensorOptions &options) {
    if (model_config.is_tx_model()) {
        return load_tx_model(model_config, options);
    }
    return load_lstm_model(model_config, options);
}

size_t auto_calculate_num_runners(const CRFModelConfig &model_config, float memory_fraction) {
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
