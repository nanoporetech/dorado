#include "encode_kmer.h"

#include "utils/sequence_utils.h"
#include "utils/simd.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// OneHot encoding encodes categorical data (bases) into unique bool-like columns for each category.
// This function returns a u32 representing the 4 i8 base categories and a 5th for N.
// -1(N)[0,0,0,0]; 0(A)[1,0,0,0]; 1(C)[0,1,0,0]; 2(G)[0,0,1,0] 3(T)[0,0,0,1].
// The encoding is done by bit shifting a 1 by the numerical "magnitude" of the base giving:
// Note: Bytes [ABCD] becomes [D,C,B,A] in LE systems which is why T is [0,0,0,1] not [1,0,0,0].
inline uint32_t encode(int base) { return base == -1 ? uint32_t{0} : (uint32_t{1} << (base << 3)); }

// Write the kmer encoding into `output_ptr` whose size must be at least: `kmer_len * 4 * context_samples`
inline void encode_kmer_generic(int8_t* output_ptr,
                                const std::vector<int>& seq,
                                const std::vector<uint64_t>& seq_mappings,
                                size_t context_seq_len,
                                size_t kmer_len) {
    const size_t seq_len = std::min(seq.size(), context_seq_len);
    for (size_t s = 0; s < seq_len; ++s) {
        uint64_t sample_st = seq_mappings[s];
        uint64_t sample_en = seq_mappings[s + 1];
        for (size_t b = sample_st; b < sample_en; ++b) {
            for (size_t k = 0; k < kmer_len; ++k) {
                const size_t seq_idx = s + k;
                assert(seq_idx < seq.size());
                uint32_t base_onehot = encode(seq[seq_idx]);
                // memcpy will be translated to a single 32 bit write.
                std::memcpy(output_ptr, &base_onehot, sizeof(base_onehot));
                output_ptr += sizeof(base_onehot);
            }
        }
    }
}

// Fallback path for non-AVX / kmer lengths not specifically optimised.
inline std::vector<int8_t> encode_kmer_context_generic(const std::vector<int>& seq,
                                                       const std::vector<uint64_t>& seq_mappings,
                                                       size_t bases_before,
                                                       size_t bases_after,
                                                       size_t context_samples) {
    const size_t context_seq_len = seq.size() - bases_before - bases_after;
    const size_t kmer_len = bases_before + bases_after + 1;
    const size_t kmer_bytes = kmer_len * dorado::utils::BaseInfo::NUM_BASES;
    const size_t output_size = kmer_bytes * context_samples;

    std::vector<int8_t> output(output_size, 0);
    int8_t* output_ptr = &output[0];
    encode_kmer_generic(output_ptr, seq, seq_mappings, context_seq_len, kmer_len);
    return output;
}

#if ENABLE_AVX2_IMPL
__attribute__((target("avx2"))) void avx2_encode_kmer_len9(
        std::byte* output_t_ptr,
        const std::vector<int>& seq,
        const std::vector<uint64_t>& seq_mappings,
        size_t seq_len) {
    const __m256i kOnes = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);

    // Permutations for rotations of 32 bit elements by 1, 2 and 3 elements.
    const __m256i kRotate1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    const __m256i kRotate2 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256i kRotate3 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);

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
}
#endif

// For non-AVX we use the generic path that handles any kmer length.
#if ENABLE_AVX2_IMPL
__attribute__((target("default")))
#endif
std::vector<int8_t>
encode_kmer_context_len9(const std::vector<int>& seq,
                         const std::vector<uint64_t>& seq_mappings,
                         size_t bases_before,
                         size_t bases_after,
                         size_t context_samples) {
    return encode_kmer_context_generic(seq, seq_mappings, bases_before, bases_after,
                                       context_samples);
}

#if ENABLE_AVX2_IMPL
[[maybe_unused]] __attribute__((target("avx2"))) std::vector<int8_t> encode_kmer_context_len9(
        const std::vector<int>& seq,
        const std::vector<uint64_t>& seq_mappings,
        int bases_before,
        int bases_after,
        int context_samples) {
    // These cannot change without a rewrite.
    constexpr int kKmerLen = 9;
    constexpr int kNumBases = 4;
    constexpr int kKmerBytes = kKmerLen * kNumBases;

    const size_t seq_len = seq.size() - bases_before - bases_after;

    const size_t output_size = kKmerBytes * context_samples;
    std::vector<int8_t> output_t(output_size);
    std::byte* output_t_ptr = reinterpret_cast<std::byte*>(&output_t[0]);

    avx2_encode_kmer_len9(output_t_ptr, seq, seq_mappings, seq_len);
    return output_t;
}
#endif

// Fallback path for non-AVX / kmer lengths not specifically optimised.
inline std::vector<int8_t> encode_kmer_chunk_generic(const std::vector<int>& seq,
                                                     const std::vector<uint64_t>& seq_mappings,
                                                     size_t kmer_len,
                                                     size_t context_samples,
                                                     size_t padding_samples,
                                                     bool kmer_centered) {
    // Given sequence: ACGTAC
    // Uncentered 7mer: [ACGTACnnnnn] -> ACGTACn CGTACnn GTACnnn TACnnnn ACnnnnn Cnnnnnn
    // Centered 7mer:   [nnnACGTACnnn]-> nnnACGT nnACGTA nACGTAC ACGTACn CGTACnn GTACnnn
    // Extend the sequence with N bases but do not change the mapping so the signal alignment
    // remains unchanged. Offset the copy by start_pos to center the kmer.
    const size_t start_pos = kmer_centered ? kmer_len / 2 : 0;
    std::vector<int> ext_seq(seq.size() + kmer_len - 1, -1);
    std::copy(seq.begin(), seq.end(), ext_seq.begin() + start_pos);

    const size_t kmer_bytes = kmer_len * dorado::utils::BaseInfo::NUM_BASES;
    const size_t total_samples = context_samples + (2 * padding_samples);
    const size_t output_size = kmer_bytes * total_samples;
    const size_t padded_start = kmer_bytes * padding_samples;

    std::vector<int8_t> output(output_size, 0);
    int8_t* output_ptr = &output[padded_start];

    encode_kmer_generic(output_ptr, ext_seq, seq_mappings, seq.size(), kmer_len);
    return output;
}

// For non-AVX we use the generic path that handles any kmer length.
#if ENABLE_AVX2_IMPL
__attribute__((target("default")))
#endif
std::vector<int8_t>
encode_kmer_chunk_len9(const std::vector<int>& seq,
                       const std::vector<uint64_t>& seq_mappings,
                       size_t context_samples,
                       size_t padding_samples,
                       bool kmer_centered) {
    constexpr size_t kKmerLen = 9;
    return encode_kmer_chunk_generic(seq, seq_mappings, kKmerLen, context_samples, padding_samples,
                                     kmer_centered);
}

#if ENABLE_AVX2_IMPL
__attribute__((target("avx2"))) std::vector<int8_t> encode_kmer_chunk_len9(
        const std::vector<int>& seq,
        const std::vector<uint64_t>& seq_mappings,
        size_t context_samples,
        size_t padding_samples,
        bool kmer_centered) {
    // These cannot change without a rewrite.
    constexpr int kKmerLen = 9;
    constexpr int kNumBases = 4;

    const size_t start_pos = kmer_centered ? kKmerLen / 2 : 0;
    std::vector<int> ext_seq(seq.size() + kKmerLen - 1, -1);
    std::copy(seq.begin(), seq.end(), ext_seq.begin() + start_pos);

    constexpr int kKmerBytes = kKmerLen * kNumBases;
    const size_t total_samples = context_samples + (2 * padding_samples);
    const size_t output_size = kKmerBytes * total_samples;
    const size_t padded_start = kKmerBytes * padding_samples;
    std::vector<int8_t> output_t(output_size);
    std::byte* output_t_ptr = reinterpret_cast<std::byte*>(&output_t[padded_start]);

    avx2_encode_kmer_len9(output_t_ptr, ext_seq, seq_mappings, seq.size());
    return output_t;
}
#endif

}  // namespace

namespace dorado::modbase {

std::vector<int8_t> encode_kmer_context(const std::vector<int>& seq,
                                        const std::vector<uint64_t>& seq_mappings,
                                        size_t bases_before,
                                        size_t bases_after,
                                        size_t context_samples) {
    // Specialised version for the case of kmer_len 9 that can be faster.
    const size_t kmer_len = bases_before + bases_after + 1;
    if (kmer_len == 9) {
        return encode_kmer_context_len9(seq, seq_mappings, bases_before, bases_after,
                                        context_samples);
    }
    return encode_kmer_context_generic(seq, seq_mappings, bases_before, bases_after,
                                       context_samples);
}

// Encodes a kmer chunk of the length `context_samples` symmetrically extending the
// ends by `padding_samples` kmers of 'N' bases. Optionally centers the kmer if `kmer_centered` is set.
// The returned vector is size `4 * kmer_len * (context_samples + 2*padding_samples)`
std::vector<int8_t> encode_kmer_chunk(const std::vector<int>& seq,
                                      const std::vector<uint64_t>& seq_mappings,
                                      size_t kmer_len,
                                      size_t context_samples,
                                      size_t padding_samples,
                                      bool kmer_centered) {
    if (kmer_len == 9) {
        return encode_kmer_chunk_len9(seq, seq_mappings, context_samples, padding_samples,
                                      kmer_centered);
    }
    return encode_kmer_chunk_generic(seq, seq_mappings, kmer_len, context_samples, padding_samples,
                                     kmer_centered);
}

}  // namespace dorado::modbase
