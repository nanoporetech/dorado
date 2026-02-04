#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::secondary {

struct Sample {
    int32_t seq_id = -1;
    at::Tensor features;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    at::Tensor depth;
    std::vector<std::string> read_ids_left;
    std::vector<std::string> read_ids_right;

    int64_t start() const { return (std::empty(positions_major) ? -1 : (positions_major.front())); }

    int64_t end() const {
        return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
    }

    std::pair<int64_t, int64_t> get_position(const int64_t idx) const {
        if ((idx < 0) || (idx >= static_cast<int64_t>(std::size(positions_major)))) {
            return {-1, -1};
        }
        return {positions_major[idx], positions_minor[idx]};
    }

    std::pair<int64_t, int64_t> get_last_position() const {
        return get_position(static_cast<int64_t>(std::size(positions_major)) - 1);
    }

    int64_t find_max_depth(int64_t start_idx, int64_t end_idx) const;

    void validate() const;
};

Sample slice_sample(const Sample& sample,
                    const int64_t idx_start,
                    const int64_t idx_end,
                    const bool clone);

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end);

void merge_adjacent_samples_in_place(Sample& lh, const Sample& rh);

void debug_print_sample(std::ostream& os,
                        const Sample& sample,
                        int64_t start /*= 0*/,
                        int64_t end /*= -1 */,
                        bool debug /*= false */);

std::ostream& operator<<(std::ostream& os, const Sample& sample);

std::string sample_to_string(const Sample& sample);

}  // namespace dorado::secondary
