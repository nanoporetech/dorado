#pragma once

#include "Decoder.h"

#include <torch/torch.h>

namespace dorado {

class GPUDecoder : Decoder {
public:
    std::vector<DecodedChunk> beam_search(const torch::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options) final;
    constexpr static torch::ScalarType dtype = torch::kF16;

    // We split beam_search into two parts, the first one running on the GPU and the second
    // one on the CPU. While the second part is running we can submit more commands to the GPU
    // on another thread.
    torch::Tensor gpu_part(torch::Tensor scores, int num_chunks, DecoderOptions options);
    std::vector<DecodedChunk> cpu_part(torch::Tensor moves_sequence_qstring_cpu);

private:
    torch::Tensor chunks;
    torch::Tensor chunk_results;
    torch::Tensor aux;
    torch::Tensor path;
    torch::Tensor moves_sequence_qstring;
    bool initialized{false};
};

}  // namespace dorado
