#pragma once

#include "ModBaseModelUtils.h"
#include "config/ModBaseModelConfig.h"
#include "nn/LinearUpsample.h"

#include <torch/nn.h>

#include <filesystem>
#include <vector>

#if DORADO_CUDA_BUILD
#include "nn/ConvStack.h"
#include "nn/LSTMStack.h"
#endif

namespace dorado::modbase::model {

struct ModBaseConvLSTMV3ModelImpl : torch::nn::Module {
    explicit ModBaseConvLSTMV3ModelImpl(const config::modbase::ConvLSTMV3ModelConfig& config);

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir);

    const config::modbase::ConvLSTMV3ModelConfig m_config;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    torch::nn::LSTM lstm1{nullptr};
    torch::nn::LSTM lstm2{nullptr};
    torch::nn::Linear linear{nullptr};
    nn::LinearUpsample upsample{nullptr};
};

TORCH_MODULE(ModBaseConvLSTMV3Model);

#if DORADO_CUDA_BUILD

struct ModBaseConvLSTMV3CUDAModelImpl : torch::nn::Module {
    ModBaseConvLSTMV3CUDAModelImpl(const config::modbase::ConvLSTMV3ModelConfig& config,
                                   int batch_size);

    at::Tensor forward(const at::Tensor& sigs_N1T, const at::Tensor& seqs_NTC);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir);

    const config::modbase::ConvLSTMV3ModelConfig m_config;

    nn::ConvStack sig_convs{nullptr};
    nn::ConvStack seq_convs{nullptr};
    nn::ConvStack merge_conv{nullptr};
    nn::LSTMStack lstm{nullptr};
    torch::nn::Linear linear{nullptr};
    nn::LinearUpsample upsample{nullptr};

    const int batch_size;
    nn::WorkingMemory wm{batch_size}, wm_seq{batch_size};
};

TORCH_MODULE(ModBaseConvLSTMV3CUDAModel);

#endif

}  // namespace dorado::modbase::model
