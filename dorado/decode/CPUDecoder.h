#pragma once

#include "Decoder.h"

#include <torch/torch.h>

namespace dorado {

class CPUDecoder final : Decoder {
public:
    std::vector<DecodedChunk> beam_search(const torch::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options) final;
    constexpr static torch::ScalarType dtype = torch::kF32;
};

}  // namespace dorado
