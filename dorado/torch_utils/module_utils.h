#pragma once

#include "torch_utils/tensor_utils.h"

#include <spdlog/spdlog.h>
#include <torch/nn.h>
#include <torch/script.h>

#include <cassert>
#include <stdexcept>
#include <variant>
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

// Simple wrapper for one of
// 1) torch::nn::ModuleHolder<torch::nn::AnyModule>
// 2) torch::jit::Module
// providing a consistent forward interface.
class ModuleWrapper {
public:
    ModuleWrapper() : m_module(std::monostate()) {}
    ModuleWrapper(torch::nn::ModuleHolder<torch::nn::AnyModule>&& nn_module)
            : m_module(std::move(nn_module)) {}
    ModuleWrapper(torch::jit::Module&& jit_module) : m_module(std::move(jit_module)) {}

    // Currently assumes there is a single output tensor.
    template <typename... Inputs>
    at::Tensor forward(Inputs&&... inputs) {
        if (std::holds_alternative<torch::nn::ModuleHolder<torch::nn::AnyModule>>(m_module)) {
            auto& nn_module = std::get<torch::nn::ModuleHolder<torch::nn::AnyModule>>(m_module);
            return nn_module->forward(std::forward<Inputs>(inputs)...);
        } else if (std::holds_alternative<torch::jit::Module>(m_module)) {
            auto& jit_module = std::get<torch::jit::Module>(m_module);
            std::vector<torch::jit::IValue> jit_inputs{std::forward<Inputs>(inputs)...};
            return jit_module.forward(jit_inputs).toTensor();
        } else {
            throw std::runtime_error("Forward called on invalid module");
        }
    }

private:
    std::variant<std::monostate, torch::nn::ModuleHolder<torch::nn::AnyModule>, torch::jit::Module>
            m_module;
};

}  // namespace dorado::utils
