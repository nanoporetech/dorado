#pragma once

#include "config/ModBaseModelConfigFwd.h"
#include "torch_utils/module_utils.h"
#include "utils/types.h"

#include <torch/nn.h>

#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace dorado::modbase {

using ModBaseLatentSpaceModels =
        std::tuple<torch::nn::ModuleHolder<torch::nn::AnyModule>,
                   torch::nn::ModuleHolder<torch::nn::AnyModule>,
                   std::vector<torch::nn::ModuleHolder<torch::nn::AnyModule>>>;

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const config::ModBaseModelConfig& config,
        const at::TensorOptions& options,
        const int batchsize);

// Returns the sequence model, common model, and classifier models in config order, or std::nullopt
// when configs is empty. `max_seq_len` is the number of timesteps in a basecall chunk's embeddings.
std::optional<ModBaseLatentSpaceModels> load_modbase_latent_space_model(
        std::span<const config::ModBaseModelConfig> configs,
        const at::TensorOptions& options,
        int max_seq_len);

// Alphabet, long names and motif context of the given latent-space modbase models.
std::shared_ptr<ModBaseInfo> create_modbase_info(
        std::span<const config::ModBaseModelConfig> configs);

// Runs the sequence, common and classifier models on the [3, N, T] int8 decoder output and the
// basecall model's signal embeddings. Returns one [N, T * num_states] tensor per classifier head.
std::vector<at::Tensor> run_modbase_latent_space_models(ModBaseLatentSpaceModels& models,
                                                        const at::Tensor& decoded_data,
                                                        at::Tensor signal_embeddings);

}  // namespace dorado::modbase
