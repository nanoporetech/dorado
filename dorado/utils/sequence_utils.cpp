#include "utils/sequence_utils.h"

#include "utils/log_utils.h"
#include "utils/simd.h"

#include <edlib.h>
#include <minimap.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <optional>
#include <vector>

namespace {

#if ENABLE_AVX2_IMPL
__attribute__((target("default")))
#endif
std::string
reverse_complement_impl(const std::string& sequence) {
    if (sequence.empty()) {
        return {};
    }

    const auto num_bases = sequence.size();
    std::string rev_comp_sequence;
    rev_comp_sequence.resize(num_bases);

    // Run every template base through the table, reading in reverse order.
    const char* template_ptr = &sequence[num_bases - 1];
    char* complement_ptr = &rev_comp_sequence[0];
    for (size_t i = 0; i < num_bases; ++i) {
        const auto template_base = *template_ptr--;
        *complement_ptr++ = dorado::utils::complement_table[template_base];
    }
    return rev_comp_sequence;
}

#if ENABLE_AVX2_IMPL
// AVX2 implementation that does in-register lookups of 32 bases at once, using
// PSHUFB. On strings with over several thousand bases this was measured to be about 10x the speed
// of the default implementation on Skylake.
__attribute__((target("avx2"))) std::string reverse_complement_impl(const std::string& sequence) {
    const auto len = sequence.size();
    std::string rev_comp_sequence;
    rev_comp_sequence.resize(len);

    // Maps from lower 4 bits of template base ASCII to complement base ASCII.
    // It happens that the low 4 bits of A, C, G and T ASCII encodings are unique, and
    // these are the only bits the PSHUFB instruction we use cares about (aside from the high
    // bit, which won't be set for valid input).
    // 'A' & 0xf = 1
    // 'C' & 0xf = 3
    // 'T' & 0xf = 4
    // 'G' & 0xf = 7
    // The lowest 4 bits are the same for upper and lower case, so the lookup still works for
    // lower case, but the results will always be upper case.
    const __m256i kComplementTable =
            _mm256_setr_epi8(0, 'T', 0, 'G', 'A', 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0, 0, 'T', 0, 'G',
                             'A', 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0);

    // PSHUFB indices to reverse bytes within a 16 byte AVX lane.  Note that _mm256_set_..
    // intrinsics have a high to low ordering.
    const __m256i kByteReverseTable =
            _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
                            6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    // Mask for upper / lower case bits: if set, the character is lower case.
    const __m256i kCaseBitMask = _mm256_set1_epi8(0x20);

    // Unroll to AVX register size.  Unrolling further would probably help performance.
    static constexpr size_t kUnroll = 32;

    // This starts pointing at the beginning of the first complete 32 byte template chunk
    // that we load -- i.e. the one last in memory.
    const char* template_ptr = &sequence[len - kUnroll];
    char* complement_ptr = &rev_comp_sequence[0];

    // Main vectorised loop: 32 bases per iteration.
    for (size_t chunk_i = 0; chunk_i < len / kUnroll; ++chunk_i) {
        // Load template bases.
        const __m256i template_bases =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(template_ptr));
        // Extract the bit that signifies upper / lower case.
        const __m256i case_bits = _mm256_and_si256(template_bases, kCaseBitMask);
        // Look up complement bases as upper case (where the case bit is not set).
        const __m256i complement_bases_upper_case =
                _mm256_shuffle_epi8(kComplementTable, template_bases);
        // Reinstate bits signifying lower case.
        const __m256i complement_bases = _mm256_or_si256(complement_bases_upper_case, case_bits);
        // Reverse byte order within 16 byte AVX lanes.
        const __m256i reversed_lanes = _mm256_shuffle_epi8(complement_bases, kByteReverseTable);
        // We store reversed lanes in reverse order to reverse 32 bytes overall.
        // We could alternatively use VPERMQ and a 256 bit store, but the shuffle
        // execution port (i.e. port 5 on Skylake) is oversubscribed.
        const __m128i upper_lane = _mm256_extracti128_si256(reversed_lanes, 1);
        const __m128i lower_lane = _mm256_castsi256_si128(reversed_lanes);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(complement_ptr), upper_lane);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(complement_ptr + 16), lower_lane);

        template_ptr -= kUnroll;
        complement_ptr += kUnroll;
    }

    // Loop for final 0-31 chars.
    const size_t remaining_len = len % kUnroll;
    const __m256i kZero = _mm256_setzero_si256();
    template_ptr = &sequence[remaining_len - 1];
    for (size_t i = 0; i < remaining_len; ++i) {
        // Same steps as in the main loop, but char by char, so there's no
        // reversal of byte ordering, and we load/store with scalar instructions.
        const __m256i template_base = _mm256_insert_epi8(kZero, *template_ptr--, 0);
        const __m256i case_bit = _mm256_and_si256(template_base, kCaseBitMask);
        const __m256i complement_base_upper_case =
                _mm256_shuffle_epi8(kComplementTable, template_base);
        const __m256i complement_base = _mm256_or_si256(complement_base_upper_case, case_bit);
        *complement_ptr++ = _mm256_extract_epi8(complement_base, 0);
    }

    return rev_comp_sequence;
}
#endif

}  // namespace

namespace dorado::utils {

size_t find_rna_polya(const std::string& seq) {
    // Number of bases to search at the end of the RNA sequence
    const size_t kSearchSize = 200;
    // Minimum contiguous length of a polyA
    const size_t kMinPolyASize = 5;

    const size_t size = seq.size();
    const size_t end = (kSearchSize < size) ? size - kSearchSize : size_t(0);
    size_t polya_size = 0;
    size_t polya_end_idx = size;

    // In RNA - sequence is reversed
    for (size_t i = size; i > end; --i) {
        if (seq[i - 1] == 'A') {
            if (++polya_size >= kMinPolyASize) {
                polya_end_idx = i - 1;
            }
        } else if (polya_end_idx != size) {
            break;
        } else {
            polya_size = 0;
        }
    }

    return polya_end_idx;
}

float mean_qscore_from_qstring(std::string_view qstring) {
    if (qstring.empty()) {
        return 0.0f;
    }

    // Lookup table avoids repeated invocation of std::pow, which
    // otherwise dominates run time of this function.
    // Unfortunately std::pow is not constexpr, so this can't be.
    static const auto kCharToScoreTable = [] {
        std::array<float, 256> a{};
        for (int q = 33; q <= 127; ++q) {
            auto shifted = static_cast<float>(q - 33);
            a[q] = std::pow(10.0f, -shifted / 10.0f);
        }
        return a;
    }();
    float total_error =
            std::accumulate(qstring.cbegin(), qstring.cend(), 0.0f,
                            [](float sum, char qchar) { return sum + kCharToScoreTable[qchar]; });
    float mean_error = total_error / static_cast<float>(qstring.size());
    float mean_qscore = -10.0f * std::log10(mean_error);
    return std::clamp(mean_qscore, 1.0f, 50.0f);
}

std::vector<int> sequence_to_ints(const std::string& sequence) {
    NVTX3_FUNC_RANGE();
    std::vector<int> sequence_ints;
    sequence_ints.reserve(sequence.size());
    std::transform(std::begin(sequence), std::end(sequence), std::back_inserter(sequence_ints),
                   &base_to_int);
    return sequence_ints;
}

int64_t sequence_to_move_table_index(const std::vector<uint8_t>& move_vals,
                                     int64_t sequence_index,
                                     int64_t sequence_size) {
    const int64_t moves_sz = static_cast<int64_t>(move_vals.size());
    // Check out-of-bounds and input consistency
    const bool oob_moves = sequence_index >= moves_sz;
    const bool oob_seq = sequence_index >= sequence_size;
    const bool size_invalid = sequence_size > moves_sz;

    if (move_vals.empty() || oob_moves || oob_seq || size_invalid) {
        trace_log(
                "sequence_to_move_table_index - bad input "
                "seq_index:{} seq_size:{} move.size:{} - reason empty_moves: {} "
                "oob_moves: {} oob_seq {} size_invalid: {}",
                sequence_index, sequence_size, moves_sz, move_vals.empty(), oob_moves, oob_seq,
                size_invalid);
        return -1;
    }

    if (sequence_index <= sequence_size / 2) {
        // Start with -1 because as soon as the first move_val==1 is encountered,
        // we have moved to the first base.
        int64_t seq_base_pos = -1;
        for (int64_t i = 0; i < moves_sz; i++) {
            if (move_vals[i] == 1) {
                seq_base_pos++;
                // seq_base_pos always > 0
                if (seq_base_pos == sequence_index) {
                    return i;
                }
            }
        }
    } else {
        // Start with size because as soon as the first move_val==1 is encountered,
        // we have moved to the last index (size - 1).
        int64_t seq_base_pos = sequence_size;
        for (int64_t i = moves_sz - 1; i >= 0; --i) {
            if (move_vals[i] == 1) {
                seq_base_pos--;
                if (seq_base_pos == sequence_index) {
                    return i;
                }
            }
        }
    }
    return -1;
}

// Convert a move table to an array of the indices of the start/end of each base in the signal
std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                   size_t block_stride,
                                   size_t signal_len,
                                   std::optional<size_t> reserve_size) {
    NVTX3_FUNC_RANGE();
    std::vector<uint64_t> seq_to_sig_map;
    if (reserve_size) {
        seq_to_sig_map.reserve(*reserve_size);
    }

    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == 1) {
            seq_to_sig_map.push_back(i * block_stride);
        }
    }
    seq_to_sig_map.push_back(signal_len);
    return seq_to_sig_map;
}

std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves) {
    std::vector<uint64_t> ans(moves.size(), 0);
    if (!moves.empty()) {
        ans[0] = moves[0];
    }
    for (size_t i = 1, n = moves.size(); i < n; i++) {
        ans[i] = ans[i - 1] + moves[i];
    }
    return ans;
}

// Reverse sequence to signal map inplace
void reverse_seq_to_sig_map(std::vector<uint64_t>& seq_to_sig_map, size_t signal_len) {
    // This function performs the following in 1 iteration instead of 2
    // std::reverse(v.begin(), v.end());
    // std::transform(v.begin(), v.end(), v.begin(), [](auto a) { return L - a; });

    const size_t end_idx = seq_to_sig_map.size();
    const size_t mid_idx = end_idx / 2;
    for (size_t left_idx = 0; left_idx < mid_idx; ++left_idx) {
        const size_t right_idx = end_idx - left_idx - 1;
        auto& left = seq_to_sig_map[left_idx];
        auto& right = seq_to_sig_map[right_idx];
        assert(signal_len >= left);
        assert(signal_len >= right);
        left = signal_len - left;
        right = signal_len - right;
        std::swap(left, right);
    }

    // Handle the middle element for odd sized containers
    if (end_idx % 2 != 0) {
        auto& mid = seq_to_sig_map[mid_idx];
        assert(signal_len >= mid);
        mid = signal_len - mid;
    }
}

// Multiversioned function dispatch doesn't work across the dorado_lib linking
// boundary.  Without this wrapper, AVX machines still only execute the default
// version.
std::string reverse_complement(const std::string& sequence) {
    NVTX3_FUNC_RANGE();
    return reverse_complement_impl(sequence);
}

const std::vector<int> BaseInfo::BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

// Utility function for counting number of trailing bases of a particular type
// in a given read.
size_t count_trailing_chars(const std::string_view seq, char c) {
    if (auto pos = seq.find_last_not_of(c); pos != std::string::npos) {
        return seq.size() - pos - 1;
    }
    return seq.size();
}

// Utility function for counting number of leading bases of a particular type
// in a given read.
size_t count_leading_chars(const std::string_view seq, char c) {
    if (auto pos = seq.find_first_not_of(c); pos != std::string::npos) {
        return pos;
    }
    return seq.size();
}

}  // namespace dorado::utils
