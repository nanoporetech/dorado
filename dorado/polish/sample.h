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
};

}  // namespace dorado::polisher
