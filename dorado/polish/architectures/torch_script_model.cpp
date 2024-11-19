#include "torch_script_model.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <stdexcept>

namespace dorado::polisher {

TorchScriptModel::TorchScriptModel(const std::filesystem::path& model_path) {
    try {
        spdlog::debug("Loading model from file: {}", model_path.string());
        m_module = torch::jit::load(model_path.string());
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path.string() +
                                 " with error: " + e.what());
    }
}

torch::Tensor TorchScriptModel::forward(torch::Tensor x) {
    return m_module.forward({std::move(x)}).toTensor();
}

}  // namespace dorado::polisher
