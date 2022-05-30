#pragma once

#include "Decoder.h"

#include <torch/torch.h>

class GPUDecoder : Decoder {
public:
    std::vector<DecodedChunk> beam_search(torch::Tensor scores,
                                          int num_chunks,
                                          DecoderOptions options) final;
    constexpr static torch::ScalarType dtype = torch::kF16;

private:
    torch::Tensor chunks;
    torch::Tensor chunk_results;
    torch::Tensor aux;
    torch::Tensor path;
    torch::Tensor moves;
    torch::Tensor sequence;
    torch::Tensor qstring;
    bool initialized{false};
    int get_cuda_device_id_from_device(const c10::Device& device);
};
