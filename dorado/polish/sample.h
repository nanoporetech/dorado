#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace dorado::polisher {

struct Sample {
    std::string ref_name;
    torch::Tensor features;
    torch::Tensor positions;
    torch::Tensor depth;
    int64_t region_start = 0;
    int64_t region_end = 0;
    int32_t seq_id = -1;
    int32_t window_id = -1;
};

}  // namespace dorado::polisher
