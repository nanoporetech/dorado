#include "torch_model_base.h"

namespace dorado::polisher {

torch::Device TorchModel::get_device() const {
    // Get the device of the first parameter as a representative.
    for (const auto& param : this->parameters()) {
        if (param.defined()) {
            return param.device();
        }
    }
    return torch::Device(torch::kCPU);
}

// Convert the model to half precision
void TorchModel::to_half() {
    this->to(torch::kHalf);
    m_half_precision = true;
}

// Sets the eval mode.
void TorchModel::set_eval() { this->eval(); }

void TorchModel::to_device(torch::Device device, bool non_blocking) {
    this->to(device, non_blocking);
}

// Predict on a batch with device and precision handling.
torch::Tensor TorchModel::predict_on_batch(torch::Tensor x) {
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
