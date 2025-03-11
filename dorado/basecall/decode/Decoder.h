#pragma once

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::decode {

struct DecodedChunk {
    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;
};

struct DecoderOptions {
    size_t beam_width = 32;
    float beam_cut = 100.0;
    float blank_score = 2.0;
    float q_shift = 0.0;
    float q_scale = 1.0;
    float temperature = 1.0;
    bool move_pad = false;
};

struct DecodeData {
    at::Tensor data;
    int num_chunks;
    DecoderOptions options;
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
                                        const config::BasecallModelConfig& config);

}  // namespace dorado::basecall::decode
