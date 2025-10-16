#pragma once

#include <ATen/core/TensorBody.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dorado::utils {

namespace detail {
[[noreturn]] void trim_throw(std::size_t size, const std::pair<int, int>& interval);
}  // namespace detail

// Default trim settings.
constexpr float DEFAULT_TRIM_THRESHOLD = 2.4f;
constexpr int DEFAULT_TRIM_WINDOW_SIZE = 40;
constexpr int DEFAULT_TRIM_MIN_ELEMENTS = 3;

// Read Trimming method (removes some initial part of the raw read).
int trim(const at::Tensor& signal, float threshold, int window_size, int min_elements);

// Trim a sequence. The interval defines the portion of the read to keep.
// Throws if given an empty string.
std::string trim_sequence(const std::string& seq, const std::pair<int, int>& trim_interval);

// Trim a vector. The interval defines the portion of the vector to keep.
template <typename Container>
Container trim_vector(const Container& container, const std::pair<int, int>& trim_interval) {
    if (container.empty()) {
        return {};
    }
    if (trim_interval.first < 0 || trim_interval.second < trim_interval.first ||
        trim_interval.first > static_cast<int>(container.size()) ||
        trim_interval.second > static_cast<int>(container.size())) {
        detail::trim_throw(container.size(), trim_interval);
    }
    const auto start = container.cbegin();
    return Container(start + trim_interval.first, start + trim_interval.second);
}

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
