#pragma once

#include <optional>
#include <string>
#include <vector>

namespace dorado::utils {

// Calculate a mean qscore from a per-base Q string.
float mean_qscore_from_qstring(const std::string& qstring);

// Convert a canonical base character (ACGT) to an integer representation (1234)
// No checking is performed on the input
int base_to_int(char c);

// Convert a sequence string to integer representation
// No checking is performed on the input
std::vector<int> sequence_to_ints(const std::string& sequence);

// Convert move table to vector of indices
std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                   size_t block_stride,
                                   size_t signal_len,
                                   std::optional<size_t> reserve_size = std::nullopt);

}  // namespace dorado::utils