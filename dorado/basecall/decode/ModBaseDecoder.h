#pragma once

#include "config/ModBaseModelConfigFwd.h"
#include "modbase/MotifMatcher.h"

#include <ATen/core/TensorBody.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace dorado::basecall::decode {

// Converts the per-head scores of the latent-space modbase models, together with the decoded
// sequence and moves of a chunk, into per-base modbase probabilities and motif hits.
class ModBaseDecoder {
public:
    explicit ModBaseDecoder(std::span<const config::ModBaseModelConfig> modbase_configs);

    bool empty() const { return m_heads.empty(); }

    // data holds one CPU tensor per classifier head for a single chunk.
    std::pair<std::vector<uint8_t>, std::vector<bool>> decode(std::span<const at::Tensor> data,
                                                              std::string_view sequence,
                                                              std::span<const uint8_t> moves) const;

private:
    struct Head {
        int32_t base_id;
        int32_t num_states;
        int32_t context_num_samples_before;
        int32_t context_num_samples_after;
        modbase::MotifMatcher matcher;
    };

    int32_t m_num_states{4};
    std::array<size_t, 4> m_base_offsets;
    std::vector<Head> m_heads;
};

}  // namespace dorado::basecall::decode
