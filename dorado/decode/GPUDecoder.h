#pragma once

#include <torch/torch.h>
#include "Decoder.h"

class GPUDecoder : Decoder {

    public:
        std::vector<DecodedChunk> beam_search(torch::Tensor scores, int num_chunks, DecoderOptions options);

    private:
        torch::Tensor chunks;
        torch::Tensor chunk_results;
        torch::Tensor aux;
        torch::Tensor path;
        torch::Tensor moves;
        torch::Tensor sequence;
        torch::Tensor qstring;
        bool initialized{false};

};
