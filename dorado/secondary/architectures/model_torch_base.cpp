#include "secondary/architectures/model_torch_base.h"

namespace dorado::secondary {

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
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->to(torch::kHalf);
    m_half_precision = true;
}

void ModelTorchBase::set_normalise(const bool val) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    m_normalise = val;
}

// Sets the eval mode.
void ModelTorchBase::set_eval() {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->eval();
}

void ModelTorchBase::to_device(torch::Device device) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->to(device);
}

// Predict on a batch with device and precision handling.
torch::Tensor ModelTorchBase::predict_on_batch(torch::Tensor x) {
    std::lock_guard<std::mutex> lock(m_mutex_write);

    x = x.to(get_device());
    if (m_half_precision) {
        x = x.to(torch::kHalf);
    }
    x = forward(std::move(x));
    if (m_half_precision) {
        x = x.to(torch::kFloat);
    }
    if (m_normalise) {
        x = torch::softmax(x, -1);
    }
    x = x.cpu();
    return x;
}

const std::unordered_set<std::string>& ModelTorchBase::get_non_persistent_buffers() const {
    return m_non_persistent_buffers;
}

void ModelTorchBase::add_nonpersistent_buffer(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    m_non_persistent_buffers.emplace(name);
}

}  // namespace dorado::secondary
