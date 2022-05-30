#ifdef __APPLE__
#pragma once

#include "../utils/metal_utils.h"
#include "Decoder.h"

#include <torch/torch.h>

class MTLDecoder : Decoder {
public:
    MTLDecoder();
    std::vector<DecodedChunk> beam_search(torch::Tensor scores,
                                          int num_chunks,
                                          DecoderOptions options) final;
    constexpr static torch::ScalarType dtype = torch::kF32;

private:
    MTL::Device* device;
    MTL::CommandQueue* command_queue;
    MTL::ComputePipelineState *scan_cps, *add_softmax_cps;
    torch::Tensor scan_idx[2][2];
};

#endif  // __APPLE__
