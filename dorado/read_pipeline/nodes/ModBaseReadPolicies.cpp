#include "read_pipeline/nodes/ModBaseReadPolicies.h"

#include "read_pipeline/base/messages/DuplexRead.h"
#include "utils/sequence_utils.h"

#include <ATen/ops/flip.h>

#include <numeric>
#include <utility>

namespace dorado::modbase {

int64_t resolve_duplex_sequence_index(const int64_t hit,
                                      const int64_t target_start,
                                      const int64_t sequence_size,
                                      const bool is_template_direction) {
    if (is_template_direction) {
        return hit + target_start;
    }
    return sequence_size - (hit + target_start + 1);
}

SimplexReadPolicy::SimplexReadPolicy(ReadCommon& read) : m_read(read) {
    m_read.base_mod_simplex_motif_hits.resize(m_read.seq.size(), false);
}

bool SimplexReadPolicy::next_strand() { return !std::exchange(m_processed, true); }

const at::Tensor& SimplexReadPolicy::signal() const { return m_read.raw_data; }
const std::string& SimplexReadPolicy::sequence() const { return m_read.seq; }
const std::vector<uint8_t>& SimplexReadPolicy::moves() const { return m_read.moves; }

void SimplexReadPolicy::mark_motif_hit(size_t strand_hit) {
    m_read.base_mod_simplex_motif_hits.at(strand_hit) = true;
}

DuplexReadPolicy::DuplexReadPolicy(DuplexRead& read, size_t canonical_stride)
        : m_read(read), m_canonical_stride(canonical_stride) {}

bool DuplexReadPolicy::next_strand() {
    while (++m_strand_index < 2) {
        m_is_template_direction = m_strand_index == 0;
        const auto& stereo_inputs = m_read.stereo_feature_inputs;

        // The complement signal is stored in the opposite orientation to the duplex sequence.
        auto simplex_signal = m_is_template_direction
                                      ? stereo_inputs.template_signal
                                      : at::flip(stereo_inputs.complement_signal, 0);
        const auto& simplex_moves = m_is_template_direction ? stereo_inputs.template_moves
                                                            : stereo_inputs.complement_moves;
        const auto simplex_sequence =
                m_is_template_direction ? stereo_inputs.template_seq
                                        : utils::reverse_complement(stereo_inputs.complement_seq);
        const auto duplex_sequence = m_is_template_direction
                                             ? m_read.read_common.seq
                                             : utils::reverse_complement(m_read.read_common.seq);

        auto [moves_offset, target_start, realigned_moves] =
                utils::realign_moves(simplex_sequence, duplex_sequence, simplex_moves);
        if (moves_offset == -1 && target_start == -1 && realigned_moves.empty()) {
            continue;
        }

        const auto signal_len = realigned_moves.size() * m_canonical_stride;
        const auto num_moves = std::accumulate(realigned_moves.begin(), realigned_moves.end(), 0);
        m_sequence = duplex_sequence.substr(target_start, num_moves);
        m_signal = simplex_signal.slice(0, moves_offset * m_canonical_stride,
                                        moves_offset * m_canonical_stride + signal_len);
        m_moves = std::move(realigned_moves);
        m_target_start = target_start;
        return true;
    }
    return false;
}

size_t DuplexReadPolicy::to_read_index(size_t strand_hit) const {
    return static_cast<size_t>(resolve_duplex_sequence_index(
            static_cast<int64_t>(strand_hit), m_target_start,
            static_cast<int64_t>(m_read.read_common.seq.size()), m_is_template_direction));
}

}  // namespace dorado::modbase
