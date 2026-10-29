#pragma once

#include "Decoder.h"
#include "ModBaseDecoder.h"
#include "config/ModBaseModelConfigFwd.h"

#include <ATen/core/TensorBody.h>

#include <span>
#include <vector>

namespace dorado::basecall::decode {

class CUDADecoder final : public Decoder {
public:
    CUDADecoder(float score_range, std::span<const config::ModBaseModelConfig> modbase_configs);

    // Decode preparation runs on the GPU and decode finalization runs on the CPU. While the
    // latter is running we can submit more commands to the GPU on another thread.
    DecoderResult prepare_decode(DecoderInput input) const override;
    std::vector<DecodedChunk> decode(const DecoderResult& result) const override;

    at::ScalarType dtype() const override { return at::ScalarType::Half; }

private:
    ModBaseDecoder m_modbase_decoder;
    float m_score_range;
};

}  // namespace dorado::basecall::decode
