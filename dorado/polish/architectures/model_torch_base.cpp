#include "model_torch_base.h"

namespace dorado::polisher {

torch::Device ModelTorchBase::get_device() const {
    // Get the device of the first parameter as a representative.
    for (const auto& param : this->parameters()) {
        if (param.defined()) {
            return param.device();
        }
    }
    return torch::Device(torch::kCPU);
}

// Convert the model to half precision
void ModelTorchBase::to_half() {
    this->to(torch::kHalf);
    m_half_precision = true;
}

// Sets the eval mode.
void ModelTorchBase::set_eval() { this->eval(); }

void ModelTorchBase::to_device(torch::Device device) { this->to(device); }

// Predict on a batch with device and precision handling.
torch::Tensor ModelTorchBase::predict_on_batch(torch::Tensor x) {
    x = x.to(get_device());
    if (m_half_precision) {
        x = x.to(torch::kHalf);
    }
    x = forward(std::move(x)).detach().cpu();
    if (m_half_precision) {
        x = x.to(torch::kFloat);
    }
    return x;
}

}  // namespace dorado::polisher
