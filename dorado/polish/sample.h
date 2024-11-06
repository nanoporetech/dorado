#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dorado::polisher {

struct Sample {
    torch::Tensor features;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    torch::Tensor depth;
    int32_t seq_id = -1;

    int64_t start() const {
        return (std::empty(positions_major) ? -1 : (positions_major.front() - 1));
    }
    int64_t end() const {
        return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
    }
};

}  // namespace dorado::polisher
