#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace dorado::utils {

// Create overlapping (== overlap) chunks of equal length (== chunk_size) and return begin
// positions, where the last chunk pair will have a bigger overlap.
//
std::vector<std::size_t> generate_chunks(std::size_t num_samples,
                                         std::size_t chunk_size,
                                         std::size_t stride,
                                         std::size_t overlap);

// Create overlapping (<= overlap) chunks of similar length (<= chunk_size) and return [begin, end>
// positions adjusted with respect to stride.
//
std::vector<std::pair<std::size_t, std::size_t>> generate_variable_chunks(std::size_t num_samples,
                                                                          std::size_t chunk_size,
                                                                          std::size_t stride,
                                                                          std::size_t overlap);

}  // namespace dorado::utils
