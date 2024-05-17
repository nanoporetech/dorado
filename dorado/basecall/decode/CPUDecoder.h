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
    DecodeData beam_search_part_1(DecodeData data) const;
    std::vector<DecodedChunk> beam_search_part_2(DecodeData data) const;

    at::ScalarType dtype() const { return at::ScalarType::Float; };
};

}  // namespace dorado::basecall::decode
