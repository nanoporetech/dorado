#include "sequence_utils.h"

#include "simd.h"

#include <edlib.h>
#include <minimap.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

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
        spdlog::trace(
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

std::optional<OverlapResult> compute_overlap(const std::string& query_seq,
                                             const std::string& query_name,
                                             const std::string& target_seq,
                                             const std::string& target_name,
                                             MmTbufPtr& working_buffer) {
    std::optional<OverlapResult> overlap_result;

    // Add mm2 based overlap check.
    mm_idxopt_t idx_opt;
    mm_mapopt_t map_opt;
    mm_set_opt(0, &idx_opt, &map_opt);
    mm_set_opt("map-hifi", &idx_opt, &map_opt);

    // Equivalent to "--cap-kalloc 100m --cap-sw-mem 50m"
    map_opt.cap_kalloc = 100'000'000;
    map_opt.max_sw_mat = 50'000'000;

    const char* seqs[] = {query_seq.c_str()};
    const char* names[] = {query_name.c_str()};
    mm_idx_t* index = mm_idx_str(idx_opt.w, idx_opt.k, 0, idx_opt.bucket_bits, 1, seqs, names);
    mm_mapopt_update(&map_opt, index);

    if (!working_buffer) {
        working_buffer = MmTbufPtr(mm_tbuf_init());
    }

    int hits = 0;
    mm_reg1_t* reg = mm_map(index, int(target_seq.length()), target_seq.c_str(), &hits,
                            working_buffer.get(), &map_opt, target_name.c_str());

    mm_idx_destroy(index);

    if (hits > 0) {
        OverlapResult result;

        auto best_map = std::max_element(
                reg, reg + hits,
                [](const mm_reg1_t& l, const mm_reg1_t& r) { return l.mapq < r.mapq; });
        result.target_start = best_map->rs;
        result.target_end = best_map->re;
        result.query_start = best_map->qs;
        result.query_end = best_map->qe;
        result.mapq = best_map->mapq;
        result.rev = best_map->rev;

        overlap_result = result;
    }

    for (int i = 0; i < hits; ++i) {
        free(reg[i].p);
    }
    free(reg);

    return overlap_result;
}

// Query is the read that the moves table is associated with. A new moves table will be generated
// Which is aligned to the target sequence.
std::tuple<int, int, std::vector<uint8_t>> realign_moves(const std::string& query_sequence,
                                                         const std::string& target_sequence,
                                                         const std::vector<uint8_t>& moves) {
    assert(static_cast<int>(query_sequence.length()) ==
           std::accumulate(moves.begin(), moves.end(), 0));

    // We are going to compute the overlap between the two reads
    MmTbufPtr working_buffer;
    const auto overlap_result =
            compute_overlap(query_sequence, "query", target_sequence, "target", working_buffer);

    // clang-tidy warns about performance-no-automatic-move if |failed_realignment| is const. It should be treated as such though.
    /*const*/ auto failed_realignment = std::make_tuple(-1, -1, std::vector<uint8_t>());
    // No overlap was computed, so return the tuple (-1, -1) and an empty vector to indicate that no move table realignment was computed
    if (!overlap_result) {
        return failed_realignment;
    }
    auto query_start = overlap_result->query_start;
    auto target_start = overlap_result->target_start;
    const auto query_end = overlap_result->query_end;
    const auto target_end = overlap_result->target_end;

    // Advance the query and target position so that their first nucleotide is identical
    while (query_sequence[target_start] != target_sequence[query_start]) {
        ++query_start;
        ++target_start;
        if (static_cast<size_t>(target_start) >= query_sequence.length() ||
            static_cast<size_t>(query_start) >= target_sequence.length()) {
            return failed_realignment;
        }
    }

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    auto target_sequence_component =
            target_sequence.substr(query_start, query_end - query_start + 1);
    auto query_sequence_component =
            query_sequence.substr(target_start, target_end - target_start + 1);

    EdlibAlignResult edlib_result = edlibAlign(
            target_sequence_component.data(), static_cast<int>(target_sequence_component.length()),
            query_sequence_component.data(), static_cast<int>(query_sequence_component.length()),
            align_config);

    // Check if alignment failed (edlib_result.startLocations is null)
    if (edlib_result.startLocations == nullptr) {
        // Free the memory allocated by edlibAlign
        edlibFreeAlignResult(edlib_result);

        // Return the tuple (-1, -1) and an empty vector to indicate that no move table realignment was computed
        return failed_realignment;
    }

    // Let's keep two cursor positions - one for the new move table which we are building, and one for the old where we track where we got to
    int new_move_cursor = 0;
    int old_move_cursor = 0;

    // First step is to advance the moves table to the start of the aligment in the query.
    int moves_found = 0;

    for (int i = 0; i < int(moves.size()); i++) {
        moves_found += moves[i];
        if (moves_found == target_start + 1) {
            break;
        }
        old_move_cursor++;
    }

    int old_moves_offset =
            old_move_cursor;  // Cursor indicating where the move table should now start

    const auto alignment_size =
            static_cast<size_t>(edlib_result.endLocations[0] - edlib_result.startLocations[0]) + 1;
    // Now that we have the alignment, we need to compute the new move table, by walking along the alignment
    std::vector<uint8_t> new_moves;
    for (size_t i = 0; i < alignment_size; i++) {
        auto alignment_entry = edlib_result.alignment[i];
        if ((alignment_entry == 0) || (alignment_entry == 3)) {  // Match or mismatch
            // Need to update the new move table and move the cursor of the old move table.
            new_moves.push_back(1);  // We have a match so we need a 1 (move)
            new_move_cursor++;
            old_move_cursor++;

            while ((old_move_cursor < int(moves.size())) && moves[old_move_cursor] == 0) {
                if (old_move_cursor < (new_move_cursor + old_moves_offset)) {
                    old_move_cursor++;
                } else {
                    // If we have a zero in the old move table, we need to add zeros to the new move table to make it up
                    new_moves.push_back(0);
                    new_move_cursor++;
                    old_move_cursor++;
                }
            }
            // Update the Query and target seq cursors
        } else if (alignment_entry == 1) {  // Insertion to target
            // If we have an insertion in the target, we need to add a 1 to the new move table, and increment the new move table cursor. the old move table cursor and new are now out of sync and need fixing.
            new_moves.push_back(1);
            new_move_cursor++;
        } else if (alignment_entry == 2) {  // Insertion to Query
            // We have a query insertion, all we need to do is add zeros to the new move table to make it up, the signal can be assigned to the leftmost nucleotide in the sequence.
            new_moves.push_back(0);
            new_move_cursor++;
            old_move_cursor++;
            while ((old_move_cursor < int(moves.size())) && moves[old_move_cursor] == 0) {
                new_moves.push_back(0);
                old_move_cursor++;
                new_move_cursor++;
            }
        }
    }

    edlibFreeAlignResult(edlib_result);

    return std::make_tuple(old_moves_offset, query_start, std::move(new_moves));
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
