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

    Sample() = default;

    Sample(const int32_t seq_id_,
           at::Tensor features_,
           std::vector<int64_t> positions_major_,
           std::vector<int64_t> positions_minor_,
           at::Tensor depth_,
           std::vector<std::string> read_ids_left_,
           std::vector<std::string> read_ids_right_)
            : seq_id{seq_id_},
              features{std::move(features_)},
              positions_major{std::move(positions_major_)},
              positions_minor{std::move(positions_minor_)},
              depth{std::move(depth_)},
              read_ids_left{std::move(read_ids_left_)},
              read_ids_right{std::move(read_ids_right_)} {}

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

    void validate() const;
};

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end);

void merge_adjacent_samples_in_place(Sample& lh, const Sample& rh);

void debug_print_sample(std::ostream& os,
                        const Sample& sample,
                        int64_t start /*= 0*/,
                        int64_t end /*= -1 */,
                        bool debug /*= false */);

std::ostream& operator<<(std::ostream& os, const Sample& sample);

}  // namespace dorado::secondary
