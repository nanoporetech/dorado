#include "model_torch_script.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <stdexcept>

namespace dorado::secondary {

ModelTorchScript::ModelTorchScript(const std::filesystem::path& model_path) {
    try {
        spdlog::debug("Loading model from file: {}", model_path.string());
        m_module = torch::jit::load(model_path.string());
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path.string() +
                                 " with error: " + e.what());
    }
}

torch::Device ModelTorchScript::get_device() const {
    // Get the device of the first parameter as a representative.
    for (const auto& param : m_module.parameters()) {
        if (param.defined()) {
            return param.device();
        }
    }
    return torch::Device(torch::kCPU);
}

torch::Tensor ModelTorchScript::forward(torch::Tensor x) {
    return m_module.forward({std::move(x)}).toTensor();
}

void ModelTorchScript::to_half() {
    this->to(torch::kHalf);
    m_module.to(torch::kHalf);
    m_half_precision = true;
}

void ModelTorchScript::set_normalise(const bool val) { m_normalise = val; }

void ModelTorchScript::set_eval() {
    this->eval();
    m_module.eval();
}

void ModelTorchScript::to_device(torch::Device device) {
    this->to(device);
    m_module.to(device);
}

}  // namespace dorado::secondary
