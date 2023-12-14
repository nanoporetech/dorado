#pragma once

#include "Decoder.h"

#include <ATen/core/TensorBody.h>

namespace dorado::basecall::decode {

class CPUDecoder final : Decoder {
public:
    std::vector<DecodedChunk> beam_search(const at::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options) final;
    constexpr static at::ScalarType dtype = at::ScalarType::Float;
};

}  // namespace dorado::basecall::decode
