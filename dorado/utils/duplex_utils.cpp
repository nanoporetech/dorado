#include "duplex_utils.h"

#include "torch/torch.h"

#include <algorithm>
#include <fstream>
#include <vector>

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
#include <immintrin.h>
#endif

namespace {

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
__attribute__((target("default")))
#endif
std::string
reverse_complement_impl(const std::string& sequence) {
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

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
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
std::map<std::string, std::string> load_pairs_file(std::string pairs_file_path) {
    std::ifstream dataFile;
    dataFile.open(pairs_file_path);

    std::map<std::string, std::string> template_complement_map;

    if (!dataFile.is_open()) {
        throw std::runtime_error("Pairs file does not exist.");
    }
    std::string cell;
    int line = 0;

    std::getline(dataFile, cell);
    while (!dataFile.eof()) {
        char delim = ' ';
        auto delim_pos = cell.find(delim);

        std::string t = cell.substr(0, delim_pos);
        std::string c = cell.substr(delim_pos + 1, delim_pos * 2 - 1);
        template_complement_map[t] = c;

        line++;
        std::getline(dataFile, cell);
    }
    return template_complement_map;
}

std::unordered_set<std::string> get_read_list_from_pairs(
        std::map<std::string, std::string> template_complement_map) {
    std::unordered_set<std::string> read_list;
    for (auto const& x : template_complement_map) {
        read_list.insert(x.first);
        read_list.insert(x.second);
    }
    return read_list;
}

// Multiversioned function dispatch doesn't work across the dorado_lib linking
// boundary.  Without this wrapper, AVX machines still only execute the default
// version.
std::string reverse_complement(const std::string& sequence) {
    return reverse_complement_impl(sequence);
}

std::pair<std::pair<int, int>, std::pair<int, int>> get_trimmed_alignment(
        int num_consecutive_wanted,
        unsigned char* alignment,
        int alignment_length,
        int target_cursor,
        int query_cursor,
        int start_alignment_position,
        int end_alignment_position) {
    int num_consecutive = 0;

    // Find forward trim.
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[start_alignment_position] != 2) {
            target_cursor++;
        }

        if (alignment[start_alignment_position] != 1) {
            query_cursor++;
        }

        if (alignment[start_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;  //reset counter
        }

        start_alignment_position++;

        if (start_alignment_position >= alignment_length) {
            break;
        }
    }

    target_cursor -= num_consecutive_wanted;
    query_cursor -= num_consecutive_wanted;

    // Find reverse trim
    num_consecutive = 0;
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[end_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;
        }

        end_alignment_position--;

        if (end_alignment_position < start_alignment_position) {
            break;
        }
    }

    start_alignment_position -= num_consecutive_wanted;
    end_alignment_position += num_consecutive_wanted;

    auto alignment_start_end = std::make_pair(start_alignment_position, end_alignment_position);
    auto query_target_cursors = std::make_pair(query_cursor, target_cursor);

    return std::make_pair(alignment_start_end, query_target_cursors);
}

// Applies a min pool filter to q scores for basespace-duplex algorithm
void preprocess_quality_scores(std::vector<uint8_t>& quality_scores, int pool_window) {
    // Apply a min-pool window to the quality scores
    auto opts = torch::TensorOptions().dtype(torch::kInt8);
    torch::Tensor t =
            torch::from_blob(quality_scores.data(), {1, (int)quality_scores.size()}, opts);
    auto t_float = t.to(torch::kFloat32);
    t.index({torch::indexing::Slice()}) =
            -torch::max_pool1d(-t_float, pool_window, 1, pool_window / 2);
}

}  // namespace dorado::utils
