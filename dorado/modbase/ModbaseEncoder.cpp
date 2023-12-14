#include "ModbaseEncoder.h"

#include "utils/sequence_utils.h"
#include "utils/simd.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace dorado::modbase {

ModBaseEncoder::ModBaseEncoder(size_t block_stride,
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

void ModBaseEncoder::init(const std::vector<int>& sequence_ints,
                          const std::vector<uint64_t>& seq_to_sig_map) {
    // gcc9 doesn't support <ranges>, which would be useful here
    m_sequence_ints = sequence_ints;
    m_sample_offsets.resize(seq_to_sig_map.size());
    for (size_t i = 0; i < seq_to_sig_map.size(); i++) {
        m_sample_offsets[i] = int(seq_to_sig_map[i]);
    }

    // last entry is the signal length
    m_signal_len = int(seq_to_sig_map.back());

    // cache sequence length
    m_seq_len = int(sequence_ints.size());
}

ModBaseEncoder::Context ModBaseEncoder::get_context(size_t seq_pos) const {
    NVTX3_FUNC_RANGE();
    if (seq_pos >= size_t(m_seq_len)) {
        throw std::out_of_range("Sequence position out of range.");
    }

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
    auto end_it = std::lower_bound(m_sample_offsets.begin(), m_sample_offsets.end(),
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
            fill_st = m_bases_before - int(seq_start);
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
                    auto val) { return val -= int(sig_start - seq_to_sig_offset); });
    chunk_seq_to_sig.front() = 0;
    chunk_seq_to_sig.back() = m_context_samples;

    context.data = encode_kmer(seq_ints, chunk_seq_to_sig);

    return context;
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

namespace {

// Fallback path for non-AVX / kmer lengths not specifically optimised.
std::vector<int8_t> encode_kmer_generic(const std::vector<int>& seq,
                                        const std::vector<int>& seq_mappings,
                                        int bases_before,
                                        int bases_after,
                                        int context_samples,
                                        int kmer_len) {
    const size_t seq_len = seq.size() - bases_before - bases_after;
    std::vector<int8_t> output(kmer_len * utils::BaseInfo::NUM_BASES * context_samples);

    int8_t* output_ptr = &output[0];
    for (size_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
        auto base_st = seq_mappings[seq_pos];
        auto base_en = seq_mappings[seq_pos + 1];

        for (int i = base_st; i < base_en; ++i) {
            for (size_t kmer_pos = 0; kmer_pos < size_t(kmer_len); ++kmer_pos) {
                auto base = seq[seq_pos + kmer_pos];
                uint32_t base_oh = (base == -1) ? uint32_t{} : (uint32_t{1} << (base << 3));
                // memcpy will be translated to a single 32 bit write.
                std::memcpy(output_ptr, &base_oh, sizeof(base_oh));
                output_ptr += 4;
            }
        }
    }
    return output;
}

// For non-AVX we use the generic path that handles any kmer length.
#if ENABLE_AVX2_IMPL
__attribute__((target("default")))
#endif
std::vector<int8_t>
encode_kmer_len9(const std::vector<int>& seq,
                 const std::vector<int>& seq_mappings,
                 int bases_before,
                 int bases_after,
                 int context_samples) {
    return encode_kmer_generic(seq, seq_mappings, bases_before, bases_after, context_samples, 9);
}

#if ENABLE_AVX2_IMPL
__attribute__((target("avx2"))) std::vector<int8_t> encode_kmer_len9(
        const std::vector<int>& seq,
        const std::vector<int>& seq_mappings,
        int bases_before,
        int bases_after,
        int context_samples) {
    // These cannot change without a rewrite.
    constexpr int kKmerLen = 9;
    constexpr int kNumBases = 4;

    const __m256i kOnes = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);

    // Permutations for rotations of 32 bit elements by 1, 2 and 3 elements.
    const __m256i kRotate1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    const __m256i kRotate2 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256i kRotate3 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);

    const size_t seq_len = seq.size() - bases_before - bases_after;
    std::vector<int8_t> output_t(kKmerLen * kNumBases * context_samples);
    std::byte* output_t_ptr = reinterpret_cast<std::byte*>(&output_t[0]);
    for (size_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
        const auto base_st = seq_mappings[seq_pos];
        const auto base_en = seq_mappings[seq_pos + 1];

        // Load the 9 base indices with 2 overlapping 256 bit loads.
        const __m256i bases_01234567 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&seq[seq_pos]));
        const __m256i bases_12345678 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&seq[seq_pos + 1]));

        // Calculate one-hot int8_t encodings within 32 bit elements by executing
        // 1 << (base_index << 3).  This is done in parallel across 8 base indices.
        // i.e. each 32 bit element one-hot encodes across 4 bases
        // -1 sequence indices will produce zero elements, which is what we want.
        const __m256i shifts_01234567 = _mm256_slli_epi32(bases_01234567, 3);
        const __m256i bases_01234567_oh = _mm256_sllv_epi32(kOnes, shifts_01234567);
        const __m256i shifts_12345678 = _mm256_slli_epi32(bases_12345678, 3);
        const __m256i bases_12345678_oh = _mm256_sllv_epi32(kOnes, shifts_12345678);

        // Permute/blend to get rotated forms of one-hot encodings.  If kKmerLen were
        // an integral power of 2 this would be far neater.  As it is, we have to
        // prerotate all these forms to get to a 128 bit store boundary.
        // TODO -- other arrangements, with one hot encoding after permuting, or int8 sequence
        // indices, might be more efficient.
        const __m256i bases_70123456_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate1);
        const __m256i bases_81234567_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate1);
        const __m256i bases_80123456_oh =
                _mm256_blend_epi32(bases_70123456_oh, bases_81234567_oh, 0x1);

        const __m256i bases_67012345_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate2);
        const __m256i bases_78123456_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate2);
        const __m256i bases_78012345_oh =
                _mm256_blend_epi32(bases_67012345_oh, bases_78123456_oh, 0x3);

        const __m256i bases_56701234_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate3);
        const __m256i bases_67812345_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate3);
        const __m256i bases_67801234_oh =
                _mm256_blend_epi32(bases_56701234_oh, bases_67812345_oh, 0x7);

        const __m128i bases_5678_oh = _mm256_extracti128_si256(bases_12345678_oh, 1);

        const int count = base_en - base_st;

        // 4x unrolled loop.
        for (int i = 0; i < count / 4; ++i) {
            // Write 4 rows of 9 uint32 elements = 144 bytes spanning 4 256 bit and 1 128 bit
            // write.  By the beginning of the 5th row we are back to base 0 at the beginning of
            // the next write.  Unrolling 8x would allow use of pure 256 bit writes, but typical
            // counts were too low for that to be a net benefit.
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 0), bases_01234567_oh);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 32), bases_80123456_oh);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 64), bases_78012345_oh);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 96), bases_67801234_oh);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output_t_ptr + 128), bases_5678_oh);
            output_t_ptr += 144;
        }

        // Remaining 0-3 iterations.
        const int remaining_count = count % 4;
        const std::uint32_t base8_oh = _mm256_extract_epi32(bases_12345678_oh, 7);
        for (int i = 0; i < remaining_count; ++i) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr), bases_01234567_oh);
            // memcpy will be translated to a single 32 bit write.
            std::memcpy(output_t_ptr + 32, &base8_oh, sizeof(base8_oh));
            output_t_ptr += 36;
        }
    }

    return output_t;
}
#endif

}  // namespace

std::vector<int8_t> ModBaseEncoder::encode_kmer(const std::vector<int>& seq,
                                                const std::vector<int>& seq_mappings) const {
    // Specialised version for the case of kmer_len 9 that can be faster.
    if (m_kmer_len == 9)
        return encode_kmer_len9(seq, seq_mappings, m_bases_before, m_bases_after,
                                m_context_samples);

    return encode_kmer_generic(seq, seq_mappings, m_bases_before, m_bases_after, m_context_samples,
                               m_kmer_len);
}

}  // namespace dorado::modbase
