#include "ModBaseDecoder.h"

#include "config/ModBaseModelConfig.h"
#include "utils/sequence_utils.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>

namespace dorado::basecall::decode {

ModBaseDecoder::ModBaseDecoder(const std::span<const config::ModBaseModelConfig> modbase_configs)
        : m_base_offsets{config::get_modbase_info(modbase_configs).base_probs_offsets()} {
    for (const config::ModBaseModelConfig &config : modbase_configs) {
        const config::modbase::ModBaseModelConfigData &modbase_data{
                config::get_model_config_data(config)};
        const config::modbase::CommonModelParams &common_params{
                config::get_common_model_params(config)};

        Head head{
                .base_id = static_cast<int32_t>(modbase_data.mods.base_id),
                .num_states = static_cast<int32_t>(modbase_data.mods.count + 1),
                .context_num_samples_before = static_cast<int32_t>(
                        modbase_data.context.samples_before / common_params.stride),
                .context_num_samples_after = static_cast<int32_t>(
                        modbase_data.context.samples_after / common_params.stride),
                .matcher = modbase::MotifMatcher{config},
        };
        m_heads.emplace_back(std::move(head));
        m_num_states += modbase_data.mods.count;
    }
}

std::pair<std::vector<uint8_t>, std::vector<bool>> ModBaseDecoder::decode(
        const std::span<const at::Tensor> data,
        const std::string_view sequence,
        const std::span<const uint8_t> moves) const {
    if (m_heads.empty()) {
        return {};
    }

    assert(data.size() == m_heads.size());

    std::vector<int32_t> sequence_to_signal;
    sequence_to_signal.reserve(sequence.size());
    for (int32_t i = 0; i < std::ssize(moves); ++i) {
        if (moves[i] == 1) {
            sequence_to_signal.emplace_back(i);
        }
    }
    assert(sequence_to_signal.size() == sequence.size());

    std::vector<uint8_t> modbase_probabilities(sequence.size() * m_num_states, 0);
    for (int32_t i = 0; i < std::ssize(sequence); ++i) {
        const int8_t base_id{utils::BaseInfo::BASE_IDS[sequence[i]]};
        assert((base_id >= 0) && (base_id <= 3));
        modbase_probabilities[(i * m_num_states) + m_base_offsets[base_id]] = 255U;
    }

    std::vector<bool> modbase_motif_hits(sequence.size(), false);

    for (int32_t i = 0; i < std::ssize(data); ++i) {
        assert(data[i].device() == at::kCPU);
        assert(data[i].dim() == 1);

        const Head &head{m_heads[i]};
        assert(data[i].size(0) == (head.num_states * std::ssize(moves)));

        // Scores are float16 from the CUDA models and float32 from the CPU models.
        const at::Tensor scores_tensor = data[i].to(at::kFloat).contiguous();
        const std::span<const float> scores(scores_tensor.data_ptr<float>(), scores_tensor.size(0));

        const size_t base_offset = m_base_offsets[head.base_id];

        const std::vector<size_t> sequence_hits{head.matcher.get_motif_hits(sequence)};
        for (const int32_t sequence_hit : sequence_hits) {
            const int32_t signal_hit = sequence_to_signal[sequence_hit];
            if (signal_hit < head.context_num_samples_before) {
                continue;
            }
            if (signal_hit > (std::ssize(moves) - head.context_num_samples_after)) {
                break;
            }
            modbase_motif_hits[sequence_hit] = true;
            for (int32_t mod_offset = 0; mod_offset < head.num_states; ++mod_offset) {
                const uint8_t score = std::min(
                        255.0f,
                        std::floor(scores[(signal_hit * head.num_states) + mod_offset] * 256));
                modbase_probabilities[(sequence_hit * m_num_states) + base_offset + mod_offset] =
                        score;
            }
        }
    }

    return {modbase_probabilities, modbase_motif_hits};
}

void ModBaseDecoder::decode(const std::span<const at::Tensor> heads,
                            const std::span<DecodedChunk> chunks) const {
    if (m_heads.empty()) {
        return;
    }
    std::vector<at::Tensor> chunk_heads(heads.size());
    for (int64_t idx = 0; idx < std::ssize(chunks); ++idx) {
        for (std::size_t i = 0; i < heads.size(); ++i) {
            chunk_heads[i] = heads[i][idx];
        }
        DecodedChunk &chunk = chunks[idx];
        std::tie(chunk.base_mod_probs, chunk.base_mod_simplex_motif_hits) =
                decode(chunk_heads, chunk.sequence, chunk.moves);
    }
}

at::Tensor pack_decoded_chunks(const std::span<const DecodedChunk> chunks) {
    const int64_t N = std::ssize(chunks);
    const int64_t T = std::ssize(chunks.front().moves);
    at::Tensor packed = at::zeros({3, N, T}, at::kChar);
    auto *const data = packed.data_ptr<int8_t>();
    for (int64_t idx = 0; idx < N; ++idx) {
        const DecodedChunk &chunk = chunks[idx];
        std::ranges::copy(chunk.moves, data + idx * T);
        std::ranges::copy(chunk.sequence, data + (N + idx) * T);
    }
    return packed;
}

}  // namespace dorado::basecall::decode
