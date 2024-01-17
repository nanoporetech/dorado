#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado::utils {

// Calculate a mean qscore from a per-base Q string.
float mean_qscore_from_qstring(std::string_view qstring);

// Convert a canonical base character (ACGT) to an integer representation (0123).
// No checking is performed on the input.
inline int base_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

// Convert a sequence string to integer representation
// No checking is performed on the input
std::vector<int> sequence_to_ints(const std::string& sequence);

// Convert move table to vector of indices
std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                   size_t block_stride,
                                   size_t signal_len,
                                   std::optional<size_t> reserve_size);

// Compute cumulative sums of the move table
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves);

// Result of overlapping two reads
using OverlapResult = std::tuple<bool, uint32_t, uint32_t, uint32_t, uint32_t>;

OverlapResult compute_overlap(const std::string& query_seq, const std::string& target_seq);

// Compute reverse complement of a nucleotide sequence.
// Bases are specified as capital letters.
// Undefined output if characters other than A, C, G, T appear.
std::string reverse_complement(const std::string& sequence);

class BaseInfo {
public:
    static constexpr int NUM_BASES = 4;
    static const std::vector<int> BASE_IDS;
};

int count_trailing_chars(const std::string_view adapter, char c);

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
