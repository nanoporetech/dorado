#pragma once

#include "basecall/DecodedChunk.h"
#include "nn/AuxiliaryData.h"

#include <ATen/core/TensorBody.h>

#include <memory>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::decode {

struct DecodeData {
    at::Tensor data;
    int num_chunks;
    DecoderOptions options;
    const nn::AuxiliaryData *aux{nullptr};
};

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual DecodeData beam_search_part_1(DecodeData data) const = 0;
    virtual std::vector<DecodedChunk> beam_search_part_2(DecodeData data) const = 0;
    // Returns the torch::TensorOptions::dtype to use for input data to models that use this decoder
    virtual at::ScalarType dtype() const = 0;
};

std::unique_ptr<Decoder> create_decoder(c10::Device device,
                                        const config::BasecallModelConfig &config);

}  // namespace dorado::basecall::decode
