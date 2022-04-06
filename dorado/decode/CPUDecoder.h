#pragma once

#include <torch/torch.h>
#include "Decoder.h"

class CPUDecoder : Decoder {

    public:
        std::vector<DecodedChunk> beam_search(torch::Tensor scores, int num_chunks, DecoderOptions options);

    private:
        torch::Tensor forward_scores(torch::Tensor scores);
        torch::Tensor backward_scores(torch::Tensor scores);
};
