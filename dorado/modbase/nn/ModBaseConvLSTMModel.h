#pragma once

#include "ModBaseModelUtils.h"
#include "config/ModBaseModelConfig.h"

#include <torch/nn.h>

#include <filesystem>
#include <vector>

namespace dorado::modbase::model {

struct ModBaseConvLSTMModelImpl : torch::nn::Module {
    explicit ModBaseConvLSTMModelImpl(const config::modbase::ConvLSTMModelConfig& config);
    explicit ModBaseConvLSTMModelImpl(const config::modbase::ConvLSTMV2ModelConfig& config);

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir);

private:
    void initialise(const config::modbase::CommonModelParams& params);

    const bool m_is_conv_lstm_v2{false};

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    torch::nn::LSTM lstm1{nullptr};
    torch::nn::LSTM lstm2{nullptr};
    torch::nn::Linear linear{nullptr};
    torch::nn::SiLU activation{nullptr};
};

TORCH_MODULE(ModBaseConvLSTMModel);

}  // namespace dorado::modbase::model
