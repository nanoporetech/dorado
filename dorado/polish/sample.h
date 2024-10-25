#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace dorado::polisher {

struct Sample {
    std::string ref_name;
    int64_t start = 0;
    int64_t end = 0;
    torch::Tensor features;
    torch::Tensor positions;
    int32_t depth = 0;
};

}  // namespace dorado::polisher
