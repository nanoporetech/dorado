#include "ModbaseEncoder.h"

#include "encode_kmer.h"
#include "utils/sequence_utils.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace dorado::modbase {

ModBaseEncoder::ModBaseEncoder(size_t block_stride,
                               size_t context_samples,
                               int bases_before,
                               int bases_after,
                               bool base_start_justified)
        : m_bases_before(bases_before),
          m_bases_after(bases_after),
          m_kmer_len(bases_before + bases_after + 1),
          m_block_stride(int(block_stride)),
          m_context_samples(int(context_samples)),
          m_context_samples_before(m_context_samples / 2),
          m_seq_len(0),
          m_signal_len(0),
          m_base_start_justified(base_start_justified) {}

void ModBaseEncoder::init(const std::vector<int>& sequence_ints,
                          const std::vector<uint64_t>& seq_to_sig_map) {
    // gcc9 doesn't support <ranges>, which would be useful here
    m_sequence_ints = sequence_ints;
    m_sample_offsets = seq_to_sig_map;

    // last entry is the signal length
    m_signal_len = int(seq_to_sig_map.back());
    assert(uint64_t(m_signal_len) == seq_to_sig_map.back());

    // cache sequence length
    m_seq_len = int(sequence_ints.size());
    assert(size_t(m_seq_len) == sequence_ints.size());
}

ModBaseEncoder::Context ModBaseEncoder::get_context(size_t seq_pos) const {
    NVTX3_FUNC_RANGE();
    if (seq_pos >= size_t(m_seq_len)) {
        throw std::out_of_range("Sequence position out of range.");
    }

    Context context{};
    int first_sample = sample_pos(int(seq_pos)) - m_context_samples_before;

    if (first_sample >= 0) {
        context.first_sample = size_t(first_sample);
        context.lead_samples_needed = 0;
    } else {
        context.first_sample = 0;
        context.lead_samples_needed = size_t(-first_sample);
    }
    int last_sample = first_sample + m_context_samples;
    if (last_sample > m_signal_len) {
        context.num_existing_samples = size_t(m_signal_len) - context.first_sample;
        context.tail_samples_needed = last_sample - m_signal_len;
    } else {
        context.num_existing_samples = size_t(last_sample) - context.first_sample;
        context.tail_samples_needed = 0;
    }

    // find base position for first and last sample
    auto start_it = std::upper_bound(m_sample_offsets.begin(), m_sample_offsets.end(),
                                     context.first_sample);
    auto end_it = std::lower_bound(m_sample_offsets.begin(), m_sample_offsets.end(),
                                   context.first_sample + context.num_existing_samples);

    auto seq_start = std::distance(m_sample_offsets.begin(), start_it) - 1;
    auto seq_end = std::distance(m_sample_offsets.begin(), end_it);

    std::vector<int> seq_ints;

    if (seq_start >= m_bases_before &&
        seq_end + m_bases_after < static_cast<int>(m_sequence_ints.size())) {
        seq_ints = {m_sequence_ints.begin() + seq_start - m_bases_before,
                    m_sequence_ints.begin() + seq_end + m_bases_after};
    } else {
        seq_ints.insert(seq_ints.end(), seq_end - seq_start + m_bases_before + m_bases_after, -1);
        auto fill_st = 0;
        auto chunk_seq_st = seq_start - m_bases_before;
        auto chunk_seq_en = seq_end + m_bases_after;
        if (seq_start < m_bases_before) {
            fill_st = m_bases_before - int(seq_start);
            chunk_seq_st = 0;
        }
        if (seq_end + m_bases_after > static_cast<int>(m_sequence_ints.size())) {
            chunk_seq_en = m_sequence_ints.size();
        }
        std::copy(m_sequence_ints.begin() + chunk_seq_st, m_sequence_ints.begin() + chunk_seq_en,
                  seq_ints.begin() + fill_st);
    }

    std::vector<uint64_t> chunk_seq_to_sig = {m_sample_offsets.begin() + seq_start,
                                              m_sample_offsets.begin() + seq_end + 1};
    std::transform(
            chunk_seq_to_sig.begin(), chunk_seq_to_sig.end(), chunk_seq_to_sig.begin(),
            [sig_start = context.first_sample, seq_to_sig_offset = context.lead_samples_needed](
                    auto val) { return val -= int(sig_start - seq_to_sig_offset); });
    chunk_seq_to_sig.front() = 0;
    chunk_seq_to_sig.back() = m_context_samples;

    context.data = encode_kmer_context(seq_ints, chunk_seq_to_sig, m_bases_before, m_bases_after,
                                       m_context_samples);
    return context;
}

int ModBaseEncoder::sample_pos(int base_pos) const {
    if (m_base_start_justified) {
        // The sample position of the context base.
        return compute_sample_pos(base_pos);
    }
    // The centroid sample beween the context base and the next base.
    return (compute_sample_pos(base_pos) + compute_sample_pos(base_pos + 1)) / 2;
}

int ModBaseEncoder::compute_sample_pos(int base_pos) const {
    int base_offset = base_pos;
    if (base_offset < 0) {
        return m_block_stride * (base_offset);
    }
    if (base_offset >= m_seq_len) {
        auto sig_len = m_signal_len;
        if (sig_len % m_block_stride != 0) {
            sig_len += m_block_stride - m_signal_len % m_block_stride;
        }
        return sig_len + m_block_stride * (base_offset - m_seq_len);
    }
    return int(m_sample_offsets[base_offset]);
}

}  // namespace dorado::modbase
