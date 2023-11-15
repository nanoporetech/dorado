#pragma once
#include <ATen/core/TensorBody.h>

namespace dorado::utils {

// Read Trimming method (removes some initial part of the raw read).
int trim(const at::Tensor& signal,
         float threshold = 2.4,
         int window_size = 40,
         int min_elements = 3);

// Trim a sequence. The interval defines the portion of the read to keep.
std::string trim_sequence(const std::string& seq, const std::pair<int, int>& trim_interval);

// Trim a quality vector. The interval defines the portion of the vector to keep.
std::vector<uint8_t> trim_quality(const std::vector<uint8_t>& qual,
                                  const std::pair<int, int>& trim_interval);

// Trim the move table. The interval defines the portion of the move table to keep.
// Returns the trimmed move table, and the number of moved trimmed from the start
// of the sequence.
std::tuple<int, std::vector<uint8_t>> trim_move_table(const std::vector<uint8_t>& move_vals,
                                                      const std::pair<int, int>& trim_interval);

// Trim the mod base info. The interval defines the portion of the read to keep.
// Returns trimmed mod base bam tag string and the mod base probabilities vector.
std::tuple<std::string, std::vector<uint8_t>> trim_modbase_info(
        const std::string& seq,
        const std::string& modbase_str,
        const std::vector<uint8_t>& modbase_probs,
        const std::pair<int, int>& trim_interval);
}  // namespace dorado::utils
