#pragma once

#include <torch/torch.h>

#include <vector>

namespace utils {

inline void load_state_dict(torch::nn::Module& module, const std::vector<torch::Tensor>& weights) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }
}

}  // namespace utils
