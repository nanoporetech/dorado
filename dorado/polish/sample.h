#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct Sample {
    torch::Tensor features;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    torch::Tensor depth;
    int32_t seq_id = -1;

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
};

}  // namespace dorado::polisher
