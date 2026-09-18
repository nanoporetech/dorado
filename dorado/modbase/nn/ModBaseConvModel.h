#pragma once

#include "ModBaseModelUtils.h"
#include "config/ModBaseModelConfig.h"

#include <torch/nn.h>

#include <filesystem>
#include <string>
#include <vector>

namespace dorado::modbase::model {

struct ModBaseConvModelImpl : torch::nn::Module {
    explicit ModBaseConvModelImpl(const config::modbase::ConvOnlyModelConfig& config);

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir);

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
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(ModBaseConvModel);

}  // namespace dorado::modbase::model
