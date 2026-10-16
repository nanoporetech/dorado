#pragma once

#include "config/ModBaseModelConfig.h"

#include <torch/nn.h>

namespace dorado::modbase::model {

/**
 * Converts the output of Decoder::prepare_decode into the one-hot encoded, signal-aligned
 * k-mers consumed by the latent-space model's sequence convolution stack.
 *
 * Input is an int8 tensor with shape [3, N, T], where the first dimension contains [moves,
 * sequence, qstring]. The sequence is the compact ASCII sequence produced by the decoder; qstring
 * is not used. Output uses NTC layout [N, T * sequence_stride, 4 * kmer_len].
 */
struct ModBaseLatentSpaceV1SequenceModelImpl : torch::nn::Module {
    explicit ModBaseLatentSpaceV1SequenceModelImpl(
            const config::modbase::LatentSpaceV1ModelConfig& config);

    at::Tensor forward(const at::Tensor& decoded_data);

    const config::modbase::LatentSpaceV1ModelConfig model_config;
};

TORCH_MODULE(ModBaseLatentSpaceV1SequenceModel);

}  // namespace dorado::modbase::model
