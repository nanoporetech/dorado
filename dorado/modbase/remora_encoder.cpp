#include "remora_encoder.h"

#include "remora_utils.h"
#include "utils/sequence_utils.h"

#include <algorithm>
#include <stdexcept>

namespace dorado {

RemoraEncoder::RemoraEncoder(size_t block_stride,
                             size_t context_samples,
                             int bases_before,
                             int bases_after)
        : m_bases_before(bases_before),
          m_bases_after(bases_after),
          m_kmer_len(bases_before + bases_after + 1),
          m_block_stride(int(block_stride)),
          m_context_samples(int(context_samples)),
          m_seq_len(0),
          m_signal_len(0) {}

void RemoraEncoder::init(const std::vector<int>& sequence_ints,
                         const std::vector<uint64_t>& seq_to_sig_map) {
    // gcc9 doesn't support <ranges>, which would be useful here
    m_sequence_ints = sequence_ints;

    // remove the final entry, we're only interested in where the bases start
    m_sample_offsets = {std::begin(seq_to_sig_map), std::prev(std::end(seq_to_sig_map), 1)};

    // last entry is the signal length
    m_signal_len = seq_to_sig_map.back();

    // cache sequence length
    m_seq_len = int(sequence_ints.size());
}

RemoraEncoder::Context RemoraEncoder::get_context(size_t seq_pos) const {
    if (seq_pos >= size_t(m_seq_len)) {
        throw std::out_of_range("Sequence position out of range.");
    }

    auto encoded_kmer_len = m_kmer_len * RemoraUtils::NUM_BASES;
    Context context{};
    int base_sample_pos =
            (compute_sample_pos(int(seq_pos)) + compute_sample_pos(int(seq_pos) + 1)) / 2;
    int samples_before = (m_context_samples / 2);
    int first_sample = base_sample_pos - samples_before;
    if (first_sample >= 0) {
        context.first_sample = size_t(first_sample);
        context.lead_samples_needed = 0;
    } else {
        context.first_sample = 0;
        context.lead_samples_needed = size_t(-first_sample);
    }
    int last_sample = first_sample + m_context_samples;
    if (last_sample > m_signal_len) {
        context.num_samples = size_t(m_signal_len) - context.first_sample;
        context.tail_samples_needed = last_sample - m_signal_len;
    } else {
        context.num_samples = size_t(last_sample) - context.first_sample;
        context.tail_samples_needed = 0;
    }

    // find base position for first and last sample
    auto start_it = std::upper_bound(m_sample_offsets.begin(), m_sample_offsets.end(),
                                     context.first_sample);
    auto end_it = std::upper_bound(m_sample_offsets.begin(), m_sample_offsets.end(),
                                   context.first_sample + context.num_samples);

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
            fill_st = m_bases_before - seq_start;
            chunk_seq_st = 0;
        }
        if (seq_end + m_bases_after > static_cast<int>(m_sequence_ints.size())) {
            chunk_seq_en = m_sequence_ints.size();
        }
        std::copy(m_sequence_ints.begin() + chunk_seq_st, m_sequence_ints.begin() + chunk_seq_en,
                  seq_ints.begin() + fill_st);
    }

    std::vector<int> chunk_seq_to_sig = {m_sample_offsets.begin() + seq_start,
                                         m_sample_offsets.begin() + seq_end + 1};
    std::transform(
            chunk_seq_to_sig.begin(), chunk_seq_to_sig.end(), chunk_seq_to_sig.begin(),
            [sig_start = context.first_sample, seq_to_sig_offset = context.lead_samples_needed](
                    auto val) { return val -= sig_start - seq_to_sig_offset; });
    chunk_seq_to_sig.front() = 0;
    chunk_seq_to_sig.back() = m_context_samples;

    context.data = encode_kmer(seq_ints, chunk_seq_to_sig);

    return context;
}

int RemoraEncoder::compute_sample_pos(int base_pos) const {
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
    return m_sample_offsets[base_offset];
}

std::vector<float> RemoraEncoder::encode_kmer(const std::vector<int>& seq,
                                              const std::vector<int>& seq_mappings) const {
    int seq_len = seq.size() - m_bases_before - m_bases_after;
    std::vector<float> output(m_kmer_len * RemoraUtils::NUM_BASES * m_context_samples);

    for (size_t kmer_pos = 0; kmer_pos < m_kmer_len; ++kmer_pos) {
        auto enc_offset = RemoraUtils::NUM_BASES * kmer_pos;
        for (size_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
            auto base = seq[seq_pos + kmer_pos];
            if (base == -1) {
                continue;
            }
            auto base_st = seq_mappings[seq_pos];
            auto base_en = seq_mappings[seq_pos + 1];
            for (size_t sig_pos = base_st; sig_pos < base_en; ++sig_pos) {
                output[m_context_samples * (enc_offset + base) + sig_pos] = 1;
            }
        }
    }
    return output;
}

}  // namespace dorado
