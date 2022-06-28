#include "remora_encoder.h"

#include "remora_utils.h"

#include <algorithm>
#include <stdexcept>

RemoraEncoder::RemoraEncoder(size_t block_stride,
                             size_t context_blocks,
                             int bases_before,
                             int bases_after)
        : m_bases_before(bases_before),
          m_kmer_len(bases_before + bases_after + 1),
          m_block_stride(int(block_stride)),
          m_context_blocks(int(context_blocks)),
          m_seq_len(0),
          m_signal_len(0) {
    m_padding = m_context_blocks / 2;
    int padding_for_bases_before = (m_kmer_len - 1 - bases_before) * int(block_stride);
    int padding_for_bases_after = (m_kmer_len - 1 - bases_after) * int(block_stride);
    int padding_for_bases = std::max(padding_for_bases_before, padding_for_bases_after);
    m_padding = std::max(padding_for_bases, m_padding);
}

void RemoraEncoder::encode_remora_data(const std::vector<uint8_t>& moves,
                                       const std::string& sequence) {
    // This code assumes that the first move value will always be 1. It also assumes that moves is only ever 0 or 1.
    m_seq_len = int(sequence.size());
    m_signal_len = int(moves.size()) * m_block_stride;
    int padded_signal_len = m_signal_len + m_block_stride * m_padding * 2;
    int encoded_data_size = padded_signal_len * m_kmer_len * RemoraUtils::NUM_BASES;
    m_sample_offsets.clear();
    m_sample_offsets.reserve(moves.size());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    m_encoded_data =
            torch::zeros({padded_signal_len, m_kmer_len * RemoraUtils::NUM_BASES}, options);

    // Note that upon initialization, encoded_data is all zeros, which corresponds to "N" characters.

    // First we need to find out which sample each base corresponds to, and make sure the moves vector is consistent
    // with the sequence length.
    int base_count = 0;
    for (int i = 0; i < int(moves.size()); ++i) {
        if (i == 0 || moves[i] == 1) {
            m_sample_offsets.push_back(i * m_block_stride);
            ++base_count;
        }
    }
    if (base_count > m_seq_len) {
        throw std::runtime_error("Movement table indicates more bases than provided in sequence (" +
                                 std::to_string(base_count) + " > " + std::to_string(m_seq_len) +
                                 ").");
    }
    if (base_count < m_seq_len) {
        throw std::runtime_error("Movement table indicates fewer bases than provided in sequence(" +
                                 std::to_string(base_count) + " < " + std::to_string(m_seq_len) +
                                 ").");
    }

    // Now we can go through each base and fill in where the 1s belong.
    for (int seq_pos = -m_kmer_len + 1; seq_pos < m_seq_len; ++seq_pos) {
        // Fill buffer with the values corresponding to the kmer that begins with the current base.
        auto buffer = torch::zeros({m_kmer_len * RemoraUtils::NUM_BASES});
        // std::fill(buffer.begin(), buffer.end(), 0.0f);
        for (int kmer_pos = 0; kmer_pos < m_kmer_len; ++kmer_pos) {
            int this_base_pos = seq_pos + kmer_pos;
            int base_offset = -1;
            if (this_base_pos >= 0 && this_base_pos < m_seq_len)
                base_offset = RemoraUtils::BASE_IDS[sequence[this_base_pos]];
            if (base_offset == -1)
                continue;
            buffer[kmer_pos * RemoraUtils::NUM_BASES + base_offset] = 1.0f;
        }

        // Now we need to copy buffer into the encoded_data vector a number of times equal to the number of samples of
        // raw data corresponding to the kmer.
        int base_sample_pos = compute_sample_pos(seq_pos + m_bases_before);
        int next_base_sample_pos = compute_sample_pos(seq_pos + m_bases_before + 1);
        int num_repeats = next_base_sample_pos - base_sample_pos;

        // This is the position in the encoded data of the first sample corresponding to the kmer that begins with the
        // current base.
        int data_pos = base_sample_pos + m_padding * m_block_stride;
        if (data_pos + num_repeats > padded_signal_len) {
            throw std::runtime_error("Insufficient padding error.");
        }
        for (int i = 0; i < num_repeats; ++i, ++data_pos) {
            m_encoded_data.index_put_({data_pos}, buffer);
        }
    }
}

RemoraEncoder::Context RemoraEncoder::get_context(size_t seq_pos) const {
    if (seq_pos >= size_t(m_seq_len)) {
        throw std::out_of_range("Sequence position out of range.");
    }
    Context context{};
    context.size = m_context_blocks * m_block_stride * m_kmer_len * RemoraUtils::NUM_BASES;
    int base_sample_pos =
            (compute_sample_pos(int(seq_pos)) + compute_sample_pos(int(seq_pos) + 1)) / 2;
    int samples_before = (m_context_blocks / 2) * m_block_stride;
    int first_sample = base_sample_pos - samples_before;
    if (first_sample >= 0) {
        context.first_sample = size_t(first_sample);
        context.lead_samples_needed = 0;
    } else {
        context.first_sample = 0;
        context.lead_samples_needed = size_t(-first_sample);
    }
    int last_sample = first_sample + m_context_blocks * m_block_stride;
    if (last_sample > m_signal_len) {
        context.num_samples = size_t(m_signal_len) - context.first_sample;
        context.tail_samples_needed = last_sample - m_signal_len;
    } else {
        context.num_samples = size_t(last_sample) - context.first_sample;
        context.tail_samples_needed = 0;
    }
    auto start_pos = m_padding * m_block_stride + first_sample;
    auto end_pos = start_pos + m_context_blocks * m_block_stride;
    context.data = m_encoded_data.index(
            {torch::indexing::Slice(start_pos, end_pos), torch::indexing::Slice()});
    return context;
}

int RemoraEncoder::compute_sample_pos(int base_pos) const {
    int base_offset = base_pos;
    if (base_offset < 0) {
        return m_block_stride * (base_offset);
    }
    if (base_offset >= m_seq_len) {
        return m_signal_len + m_block_stride * (base_offset - m_seq_len);
    }
    return m_sample_offsets[base_offset];
}
