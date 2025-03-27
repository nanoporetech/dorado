#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace dorado::secondary {

// clang-format off
struct Window {
    int32_t seq_id = -1;            // ID of the sequence from which the window was sampled.
    int64_t seq_length = 0;         // Length of the sequence where the window was sampled from.
    int64_t start = 0;              // Window start, possible overlap with neighboring windows.
    int64_t end = 0;                // Window end, possible overlap with neighboring windows.
    int64_t start_no_overlap = 0;   // Start coordinate of the unique portion of this window (no overlaps with neighbors).
    int64_t end_no_overlap = 0;     // End coordinate of the unique portion of this window (no overlaps with neighbors).
};
// clang-format on

std::ostream& operator<<(std::ostream& os, const Window& w);

bool operator==(const Window& lhs, const Window& rhs);

/**
 * \brief Linearly splits sequence lengths into windows. It also returns the backward mapping of which
 *          windows correspond to which sequences, needed for stitching.
 */
std::vector<Window> create_windows(const int32_t seq_id,
                                   const int64_t seq_start,
                                   const int64_t seq_end,
                                   const int64_t seq_len,
                                   const int32_t window_len,
                                   const int32_t window_overlap);

}  // namespace dorado::secondary
