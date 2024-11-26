#pragma once

#include "torch_utils/tensor_utils.h"

#include <spdlog/spdlog.h>
#include <torch/nn.h>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace dorado::utils {

inline void load_state_dict(torch::nn::Module& module, const std::vector<at::Tensor>& weights) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        try {
        module.parameters()[idx].data() = weights[idx].data();
        } catch (const std::exception& e) {
            spdlog::error(
                    "Failed to load state dict at module parameter {}/{}. "
                    "Check '{}' matches '{}'. Original error:'{}'",
                    idx + 1, weights.size(), print_size(module.parameters()[idx], "module"),
                    print_size(weights[idx], "input"), e.what());
            throw;
        }
    }
}

}  // namespace dorado::utils
