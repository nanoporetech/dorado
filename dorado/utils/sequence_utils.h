#pragma once

#include <optional>
#include <string>
#include <vector>

namespace dorado::utils {

// Calculate a mean qscore from a per-base Q string.
float mean_qscore_from_qstring(const std::string& qstring, int start_pos = 0);

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
                                   std::optional<size_t> reserve_size = std::nullopt);

// Compute cumulative sums of the move table
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves);

// Compute reverse complement of a nucleotide sequence.
// Bases are specified as capital letters.
// Undefined output if characters other than A, C, G, T appear.
std::string reverse_complement(const std::string& sequence);

// Convert the 4bit encoded sequence in a bam1_t structure
// into a string.
std::string convert_nt16_to_str(uint8_t* bseq, size_t slen);

}  // namespace dorado::utils
