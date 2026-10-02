#include "secondary/consensus/window_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace dorado::secondary {

std::vector<Window> create_windows(const int32_t seq_id,
                                   const int64_t seq_start,
                                   const int64_t seq_end,
                                   const int64_t seq_len,
                                   const int32_t window_len,
                                   const int32_t window_overlap,
                                   const int32_t source_region_id) {
    if (window_overlap >= (window_len / 2)) {
        DORADO_LOG(
                warn,
                "Window overlap cannot be larger than the (window_len / 2) because of transitive "
                "overlaps. Returning empty. seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                "{}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (window_len <= 0) {
        DORADO_LOG(warn,
                   "Invalid window_len given to create_windows, should be > 0. Returning empty. "
                   "seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                   "{}, window_len = {}, "
                   "window_overlap = {}",
                   seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (window_overlap < 0) {
        DORADO_LOG(
                warn,
                "Invalid window_overlap given to create_windows, should be >= 0. Returning empty. "
                "seq_id = {}, seq_start = {}, seq_end = {}, seq_len = "
                "{}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if ((seq_start < 0) || (seq_end < 0) || (seq_start >= seq_len) || (seq_end > seq_len) ||
        (seq_start >= seq_end)) {
        DORADO_LOG(
                warn,
                "Invalid start/end coordinates for creating windows. Returning empty. seq_id = {}, "
                "seq_start = {}, seq_end = {}, seq_len = {}, window_len = {}, "
                "window_overlap = {}",
                seq_id, seq_start, seq_end, seq_len, window_len, window_overlap);
        return {};
    }
    if (seq_len <= 0) {
        DORADO_LOG(warn,
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

        ret.emplace_back(
                Window{seq_id, seq_len, start, end, start_no_overlap, end, source_region_id});

        if (end == seq_end) {
            break;
        }
    }

    return ret;
}

std::vector<Window> create_windows_from_regions(
        const std::vector<Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap) {
    std::vector<Window> windows;

    for (int64_t i = 0; i < std::ssize(regions); ++i) {
        Region region = regions[i];

        DORADO_LOG(debug, "Creating windows for region: '{}'.", to_string(region));

        const auto it = draft_lookup.find(region.name);
        if (it == std::end(draft_lookup)) {
            throw std::runtime_error(
                    "Sequence specified by custom region not found in input! Sequence name: " +
                    region.name);
        }
        const auto [seq_id, seq_length] = it->second;

        region.start = std::max<int64_t>(0, region.start);
        region.end = (region.end < 0) ? seq_length : std::min(seq_length, region.end);

        if (region.start >= region.end) {
            throw std::runtime_error{"Region coordinates not valid. Given: region.name = '" +
                                     region.name +
                                     "', region.start = " + std::to_string(region.start) +
                                     ", region.end = " + std::to_string(region.end)};
        }

        std::vector<Window> new_windows =
                create_windows(static_cast<int32_t>(seq_id), region.start, region.end, seq_length,
                               bam_chunk_len, window_overlap, static_cast<int32_t>(i));

        DORADO_LOG(debug, "Generated {} windows for region: '{}'.", std::size(new_windows),
                   to_string(region));
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    return windows;
}

}  // namespace dorado::secondary
