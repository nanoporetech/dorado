#pragma once

#include "ModBaseModelUtils.h"
#include "config/ModBaseModelConfig.h"
#include "nn/ConvStack.h"
#include "nn/TxModules.h"

#include <torch/nn.h>

#include <filesystem>
#include <vector>

namespace dorado::modbase::model {

/**
 * The reusable part of a latent-space modification model.
 *
 * Both inputs use NTC layout. The returned latent-space tensor also uses NTC layout and can be
 * passed to one or more classifier models.
 */
struct ModBaseLatentSpaceV1CommonModelImpl : torch::nn::Module {
    ModBaseLatentSpaceV1CommonModelImpl(const config::modbase::LatentSpaceV1ModelConfig& config,
                                        const at::TensorOptions& options);

    at::Tensor forward(at::Tensor sequences, at::Tensor embedded_features);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) const;

    bool is_compatible(const ModBaseLatentSpaceV1CommonModelImpl& other) const;
    bool operator==(const ModBaseLatentSpaceV1CommonModelImpl& other) const;

    const config::modbase::LatentSpaceV1ModelConfig model_config;
    nn::ConvStack sequence_encoder{nullptr};
    torch::nn::Sequential projection{nullptr};
    torch::nn::Sequential common_encoder{nullptr};
};

/** The model-specific classifier applied to the output of a latent-space common model. */
struct ModBaseLatentSpaceV1ClassifierModelImpl : torch::nn::Module {
    explicit ModBaseLatentSpaceV1ClassifierModelImpl(
            const config::modbase::LatentSpaceV1ModelConfig& config);

    // Input is NTC; output logits are NCT.
    at::Tensor forward(const at::Tensor& common_output);
    void load_state_dict(const std::vector<at::Tensor>& weights);
    std::vector<at::Tensor> load_weights(const std::filesystem::path& dir) const;

    const config::modbase::LatentSpaceV1ModelConfig model_config;
    torch::nn::Sequential classifier_head{nullptr};
};

TORCH_MODULE(ModBaseLatentSpaceV1CommonModel);
TORCH_MODULE(ModBaseLatentSpaceV1ClassifierModel);

}  // namespace dorado::modbase::model
