#include "window.h"

#include <spdlog/spdlog.h>

#include <ostream>

namespace dorado::secondary {

std::ostream& operator<<(std::ostream& os, const Window& w) {
    os << "seq_id = " << w.seq_id << ", seq_length = " << w.seq_length << ", start = " << w.start
       << ", end = " << w.end << ", start_no_overlap = " << w.start_no_overlap
       << ", end_no_overlap = " << w.end_no_overlap;
    return os;
}

bool operator==(const Window& lhs, const Window& rhs) {
    return std::tie(lhs.seq_id, lhs.seq_length, lhs.start, lhs.end, lhs.start_no_overlap,
                    lhs.end_no_overlap) == std::tie(rhs.seq_id, rhs.seq_length, rhs.start, rhs.end,
                                                    rhs.start_no_overlap, rhs.end_no_overlap);
}

std::vector<Window> create_windows(const int32_t seq_id,
                                   const int64_t seq_start,
                                   const int64_t seq_end,
                                   const int64_t seq_len,
                                   const int32_t window_len,
                                   const int32_t window_overlap) {
    if (window_overlap >= (window_len / 2)) {
        spdlog::warn(
                "Window overlap cannot be larger than the (window_len / 2) because of transitive "
                "overlaps. Returning empty. seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                "{}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (window_len <= 0) {
        spdlog::warn(
                "Invalid window_len given to create_windows, should be > 0. Returning empty. "
                "seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                "{}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (window_overlap < 0) {
        spdlog::warn(
                "Invalid window_overlap given to create_windows, should be >= 0. Returning empty. "
                "seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                "{}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if ((seq_start < 0) || (seq_end < 0) || (seq_start >= seq_len) || (seq_end > seq_len) ||
        (seq_start >= seq_end)) {
        spdlog::warn(
                "Invalid start/end coordinates for creating windows. Returning empty. seq_id = {}, "
                "seq_start = {}, seq_end = {}, seq_len = {}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (seq_len <= 0) {
        spdlog::warn(
                "Invalid sequence length given to create_windows. Returning empty. seq_id = {}, "
                "seq_start = {}, seq_end = {}, seq_len = {}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }

    const int32_t num_windows =
            static_cast<int32_t>(std::ceil(static_cast<double>(seq_end - seq_start) / window_len));

    std::vector<Window> ret;
    ret.reserve(num_windows);

    int32_t win_id = 0;
    for (int64_t start = seq_start; start < seq_end;
         start += (window_len - window_overlap), ++win_id) {
        const int64_t end = std::min(seq_end, start + window_len);
        const int64_t start_no_overlap =
                (start == seq_start) ? start : std::min<int64_t>(start + window_overlap, seq_end);

        ret.emplace_back(Window{
                seq_id,
                seq_len,
                start,
                end,
                start_no_overlap,
                end,
        });

        if (end == seq_end) {
            break;
        }
    }

    return ret;
}

}  // namespace dorado::secondary
