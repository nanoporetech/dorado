#pragma once

#include "Decoder.h"

#include <torch/torch.h>

class CPUDecoder : Decoder {
public:
    std::vector<DecodedChunk> beam_search(torch::Tensor scores,
                                          int num_chunks,
                                          DecoderOptions options) final;
    constexpr static torch::ScalarType dtype = torch::kF32;

private:
    torch::Tensor forward_scores(torch::Tensor scores);
    torch::Tensor backward_scores(torch::Tensor scores);
};
