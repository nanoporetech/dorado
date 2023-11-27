#pragma once

#include "Decoder.h"

#include <ATen/core/TensorBody.h>

namespace dorado {

class GPUDecoder : Decoder {
public:
    explicit GPUDecoder(float score_clamp_val) : m_score_clamp_val(score_clamp_val) {}

    std::vector<DecodedChunk> beam_search(const at::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options) final;
    constexpr static at::ScalarType dtype = at::ScalarType::Half;

    // We split beam_search into two parts, the first one running on the GPU and the second
    // one on the CPU. While the second part is running we can submit more commands to the GPU
    // on another thread.
    at::Tensor gpu_part(at::Tensor scores, DecoderOptions options);
    std::vector<DecodedChunk> cpu_part(at::Tensor moves_sequence_qstring_cpu);

    float m_score_clamp_val;
};

}  // namespace dorado
