#pragma once

#include <torch/nn.h>

#include <vector>

namespace dorado::utils {

inline void load_state_dict(torch::nn::Module& module,
                            const std::vector<at::Tensor>& weights,
                            const std::vector<at::Tensor>& buffers,
                            const bool is_tx_model) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }

    if (!is_tx_model) {
        assert(buffers.size() == module.buffers().size());
        for (size_t idx = 0; idx < buffers.size(); idx++) {
            module.buffers()[idx].data() = buffers[idx].data();
        }
    }
}

}  // namespace dorado::utils
