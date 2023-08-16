#include "sequence_utils.h"

#include "htslib/sam.h"
#include "simd.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>

#ifdef _WIN32
// seq_nt16_str is referred to in the hts-3.lib stub on windows, but has not been declared dllimport for
//  client code, so it comes up as an undefined reference when linking the stub.
const char seq_nt16_str[] = "=ACMGRSVTWYHKDBN";
#endif  // _WIN32

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

    // Compile-time constant lookup table.
    static constexpr auto kComplementTable = [] {
        std::array<char, 256> a{};
        // Valid input will only touch the entries set here.
        a['A'] = 'T';
        a['T'] = 'A';
        a['C'] = 'G';
        a['G'] = 'C';
        return a;
    }();

    // Run every template base through the table, reading in reverse order.
    const char* template_ptr = &sequence[num_bases - 1];
    char* complement_ptr = &rev_comp_sequence[0];
    for (size_t i = 0; i < num_bases; ++i) {
        const auto template_base = *template_ptr--;
        *complement_ptr++ = kComplementTable[template_base];
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
    const __m256i kComplementTable =
            _mm256_setr_epi8(0, 'T', 0, 'G', 'A', 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0, 0, 'T', 0, 'G',
                             'A', 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0);

    // PSHUFB indices to reverse bytes within a 16 byte AVX lane.  Note that _mm256_set_..
    // intrinsics have a high to low ordering.
    const __m256i kByteReverseTable =
            _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
                            6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

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
        // Look up complement bases.
        const __m256i complement_bases = _mm256_shuffle_epi8(kComplementTable, template_bases);
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
        const __m256i complement_base = _mm256_shuffle_epi8(kComplementTable, template_base);
        *complement_ptr++ = _mm256_extract_epi8(complement_base, 0);
    }

    return rev_comp_sequence;
}
#endif

}  // namespace

namespace dorado::utils {

float mean_qscore_from_qstring(const std::string& qstring, int start_pos) {
    if (qstring.empty()) {
        return 0.0f;
    }

    if (start_pos >= qstring.length()) {
        throw std::runtime_error("Mean q-score start position (" + std::to_string(start_pos) +
                                 ") is >= length of qstring (" + std::to_string(qstring.length()) +
                                 ")");
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
            std::accumulate(qstring.cbegin() + start_pos, qstring.cend(), 0.0f,
                            [](float sum, char qchar) { return sum + kCharToScoreTable[qchar]; });
    float mean_error = total_error / static_cast<float>(qstring.size());
    float mean_qscore = -10.0f * std::log10(mean_error);
    return std::clamp(mean_qscore, 1.0f, 50.0f);
}

std::vector<int> sequence_to_ints(const std::string& sequence) {
    NVTX3_FUNC_RANGE();
    std::vector<int> sequence_ints;
    sequence_ints.reserve(sequence.size());
    std::transform(std::begin(sequence), std::end(sequence),
                   std::back_insert_iterator<std::vector<int>>(sequence_ints), &base_to_int);
    return sequence_ints;
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

// Multiversioned function dispatch doesn't work across the dorado_lib linking
// boundary.  Without this wrapper, AVX machines still only execute the default
// version.
std::string reverse_complement(const std::string& sequence) {
    NVTX3_FUNC_RANGE();
    return reverse_complement_impl(sequence);
}

std::string convert_nt16_to_str(uint8_t* bseq, size_t slen) {
    std::string seq(slen, '*');
    for (int i = 0; i < slen; i++) {
        seq[i] = seq_nt16_str[bam_seqi(bseq, i)];
    }
    return seq;
}

const std::vector<int> BaseInfo::BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

}  // namespace dorado::utils
