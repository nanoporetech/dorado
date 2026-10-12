#pragma once

#include "Decoder.h"

#include <ATen/core/TensorBody.h>

namespace dorado::basecall::decode {

namespace inner {

at::Tensor forward_scores(const at::Tensor& scores_TNC, float fixed_stay_score);
at::Tensor backward_scores(const at::Tensor& scores_TNC, float fixed_stay_score);

}  // namespace inner

class CPUDecoder final : public Decoder {
public:
    DecoderResult prepare_decode(DecoderInput input) const override;
    std::vector<DecodedChunk> decode(const DecoderResult& result) const override;

    at::ScalarType dtype() const override { return at::ScalarType::Float; }
};

}  // namespace dorado::basecall::decode
