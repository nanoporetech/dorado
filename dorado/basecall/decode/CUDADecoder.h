#pragma once

#include "Decoder.h"
#include "config/ModBaseModelConfigFwd.h"
#include "modbase/MotifMatcher.h"

#include <ATen/core/TensorBody.h>

#include <array>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace dorado::basecall::decode {

class CUDADecoder final : public Decoder {
public:
    CUDADecoder(float score_range, std::span<const config::ModBaseModelConfig> modbase_configs);

    // Decode preparation runs on the GPU and decode finalization runs on the CPU. While the
    // latter is running we can submit more commands to the GPU on another thread.
    DecoderResult prepare_decode(DecoderInput input) const override;
    std::vector<DecodedChunk> decode(const DecoderResult& result) const override;

    at::ScalarType dtype() const override { return at::ScalarType::Half; }

private:
    // part of decode
    std::pair<std::vector<uint8_t>, std::vector<bool>> decode_modbase(
            std::span<const at::Tensor> data,
            std::string_view sequence,
            std::span<const uint8_t> modes) const;

    int32_t m_modbase_num_states{4};
    std::array<size_t, 4> m_modbase_base_offsets;
    struct ModBaseDecoder {
        int32_t base_id;
        int32_t num_states;
        int32_t context_num_samples_before;
        int32_t context_num_samples_after;
        modbase::MotifMatcher matcher;
    };
    std::vector<ModBaseDecoder> m_modbase_decoders;

    float m_score_range;
};

}  // namespace dorado::basecall::decode
