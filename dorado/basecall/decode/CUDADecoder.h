#pragma once

#include "Decoder.h"

#include <ATen/core/TensorBody.h>

namespace dorado::basecall::decode {

class CUDADecoder final : public Decoder {
public:
    explicit CUDADecoder(float score_range) : m_score_range(score_range) {}

    // We split beam_search into two parts, the first one running on the GPU and the second
    // one on the CPU. While the second part is running we can submit more commands to the GPU
    // on another thread.
    DecodeData beam_search_part_1(DecodeData data) const override;
    std::vector<DecodedChunk> beam_search_part_2(const DecodeData & data) const override;

    at::ScalarType dtype() const override { return at::ScalarType::Half; }

private:
    float m_score_range;
};

}  // namespace dorado::basecall::decode
