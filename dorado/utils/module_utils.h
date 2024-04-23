#pragma once

#include <torch/nn.h>

#include <vector>

namespace dorado::utils {

inline void load_state_dict(torch::nn::Module& module, const std::vector<at::Tensor>& weights) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }
}

}  // namespace dorado::utils
