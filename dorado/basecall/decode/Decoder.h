#pragma once

#include "basecall/DecodedChunk.h"
#include "config/ModBaseModelConfig.h"
#include "nn/AuxiliaryData.h"

#include <ATen/core/TensorBody.h>

#include <memory>
#include <span>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::decode {

struct DecoderInput {
    at::Tensor scores;
    int num_chunks;
    DecoderOptions options;
    const nn::AuxiliaryData *aux{nullptr};
};

struct DecoderResult {
    at::Tensor data;
    std::vector<at::Tensor> modbase_data;  // tensor per classifier head
    int num_chunks;
    DecoderOptions options;
    const nn::AuxiliaryData *aux{nullptr};
};

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual DecoderResult prepare_decode(DecoderInput input) const = 0;
    virtual std::vector<DecodedChunk> decode(const DecoderResult &result) const = 0;
    // Returns the torch::TensorOptions::dtype to use for input data to models that use this decoder
    virtual at::ScalarType dtype() const = 0;

protected:
    Decoder() = default;
    Decoder(const Decoder &) = delete;
    Decoder &operator=(const Decoder &) = delete;
    Decoder(Decoder &&) = delete;
    Decoder &operator=(Decoder &&) = delete;
};

std::unique_ptr<Decoder> create_decoder(
        c10::Device device,
        const config::BasecallModelConfig &config,
        std::span<const config::ModBaseModelConfig> modbase_configs);

DecoderOptions create_decoder_options(const config::BasecallModelConfig &config);

}  // namespace dorado::basecall::decode
