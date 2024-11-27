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

torch::Device TorchScriptModel::get_device() const {
    // Get the device of the first parameter as a representative.
    for (const auto& param : m_module.parameters()) {
        if (param.defined()) {
            return param.device();
        }
    }
    return torch::Device(torch::kCPU);
}

torch::Tensor TorchScriptModel::forward(torch::Tensor x) {
    return m_module.forward({std::move(x)}).toTensor();
}

void TorchScriptModel::to_half() {
    this->to(torch::kHalf);
    m_module.to(torch::kHalf);
    m_half_precision = true;
}

void TorchScriptModel::set_eval() {
    this->eval();
    m_module.eval();
}

void TorchScriptModel::to_device(torch::Device device) {
    this->to(device);
    m_module.to(device);
}

}  // namespace dorado::polisher
