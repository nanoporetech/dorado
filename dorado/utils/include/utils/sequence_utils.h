#pragma once

#include "types.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado::utils {

// Returns the polyA start index in seq. The is the polyA end index in the forward direction.
// Used to trim the polyA from the qstring when calculating the mean.
size_t find_rna_polya(const std::string& seq);

// Calculate a mean qscore from a per-base Q string.
float mean_qscore_from_qstring(std::string_view qstring);

// Convert a canonical base character (ACGT) to an integer representation (0123).
// No checking is performed on the input.
inline int base_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

// Convert a sequence string to integer representation
// No checking is performed on the input
std::vector<int> sequence_to_ints(const std::string& sequence);

// Find the move table index for a given sequence index
int64_t sequence_to_move_table_index(const std::vector<uint8_t>& move_vals,
                                     int64_t sequence_index,
                                     int64_t sequence_size);

// Convert move table to vector of indices
std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                   size_t block_stride,
                                   size_t signal_len,
                                   std::optional<size_t> reserve_size);

// Compute cumulative sums of the move table
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves);

// Reverse sequence to signal map in-place
void reverse_seq_to_sig_map(std::vector<uint64_t>& seq_to_sig_map, size_t signal_len);

class BaseInfo {
public:
    static constexpr int NUM_BASES = 4;
    static const std::vector<int> BASE_IDS;
};

size_t count_trailing_chars(const std::string_view seq, char c);
size_t count_leading_chars(const std::string_view seq, char c);

/**
 * @brief Result of overlapping two reads.
 *
 * The `OverlapResult` struct holds the results of overlapping a query sequence with a target sequence.
 * The coordinates provided for `target_start`, `target_end`, `query_start`, and `query_end` indicate positions in the respective sequences.
 *
 * For example, `query_start` represents the location of the start of the query sequence in the target sequence, and so on.
 */
struct OverlapResult {
    int32_t target_start;
    int32_t target_end;
    int32_t query_start;
    int32_t query_end;
    uint8_t mapq;
    bool rev;
};
// |working_buffer| will be allocated if an empty one is passed in,
// allowing it to be reused in future calls by the caller.
std::optional<OverlapResult> compute_overlap(const std::string& query_seq,
                                             const std::string& query_name,
                                             const std::string& target_seq,
                                             const std::string& target_name,
                                             MmTbufPtr& working_buffer);

// Compute reverse complement of a nucleotide sequence.
// Bases are specified as capital letters.
// Undefined output if characters other than A, C, G, T appear.
std::string reverse_complement(const std::string& sequence);

/**
 * @brief Realigns a move table based on a given query sequence and a target sequence.
 *
 * This function adjusts the moves table to align with the target sequence, accounting
 * for any differences between the query and target sequences. It returns a tuple containing
 * an offset into the original moves table to account for trimming, a location in the target
 * sequence where the realigned sequence starts, and the newly computed move table.
 * If the new move table cannot be computed, the function returns a tuple with values (-1, -1)
 * and an empty vector.
 *
 * @param query_sequence The original sequence of moves, representing the initial alignment.
 * @param target_sequence The sequence to which the moves need to be realigned. This could
 *                        differ from the query sequence.
 * @param moves The original move table as a vector of unsigned 8-bit integers, aligned with
 *              the query sequence.
 *
 * @return std::tuple<int, int, std::vector<uint8_t>>
 *         A tuple containing:
 *         1. An offset into the old moves table (int), accounting for adjustments made during realignment.
 *         2. The start location in the target sequence (int) where the realigned sequence begins.
 *         3. The newly computed move table (std::vector<uint8_t>).
 *         If the move table cannot be computed, returns (-1, -1) and an empty vector.
 */
std::tuple<int, int, std::vector<uint8_t>> realign_moves(const std::string& query_sequence,
                                                         const std::string& target_sequence,
                                                         const std::vector<uint8_t>& moves);

// Compile-time constant lookup table.
static constexpr auto complement_table = [] {
    std::array<char, 256> a{};
    // Valid input will only touch the entries set here.
    a['A'] = 'T';
    a['T'] = 'A';
    a['C'] = 'G';
    a['G'] = 'C';
    a['a'] = 't';
    a['t'] = 'a';
    a['c'] = 'g';
    a['g'] = 'c';
    return a;
}();

}  // namespace dorado::utils
