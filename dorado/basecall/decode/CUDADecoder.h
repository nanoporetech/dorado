#pragma once

#include "Decoder.h"

#include <ATen/core/TensorBody.h>

namespace dorado::basecall::decode {

class CUDADecoder final : public Decoder {
public:
    explicit CUDADecoder(float score_range) : m_score_range(score_range) {}

    // Decode preparation runs on the GPU and decode finalization runs on the CPU. While the
    // latter is running we can submit more commands to the GPU on another thread.
    DecoderResult prepare_decode(DecoderInput input) const override;
    std::vector<DecodedChunk> decode(const DecoderResult& result) const override;

    at::ScalarType dtype() const override { return at::ScalarType::Half; }

private:
    float m_score_range;
};

}  // namespace dorado::basecall::decode
