#include "model_torch_script.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <stdexcept>

namespace dorado::secondary {

ModelTorchScript::ModelTorchScript(const std::filesystem::path& model_path) {
    try {
        spdlog::debug("Loading model from file: {}", model_path.string());
        m_module = torch::jit::load(model_path.string());
        m_module.get_method("set_normalise")({true});
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
    torch::IValue output = m_module.get_method("infer_on_features")({x});

    if (output.isTensor()) {
        // Output of the model is just a tensor.
        return output.toTensor();

    } else if (output.isTuple()) {
        // Output of the model is a tuple, return the first element.
        const auto tuple = output.toTuple();

        if (std::empty(tuple->elements())) {
            spdlog::warn("Model forward function returned an empty tuple!");
            return {};
        }

        return tuple->elements()[0].toTensor().clone();
    }

    spdlog::warn("Model returned an unsupported output type.");

    return {};
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

double ModelTorchScript::estimate_batch_memory(
        const std::vector<int64_t>& /*batch_tensor_shape*/) const {
    throw std::runtime_error{
            "Estimation of model memory consumption is not possible for the ModelTorchScript model "
            "because this is architecture specific."};
}

}  // namespace dorado::secondary
