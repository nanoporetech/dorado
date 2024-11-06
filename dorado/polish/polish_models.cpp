#include "polish_models.h"

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

GRUModel::GRUModel(const int32_t num_features,
                   const int32_t num_classes,
                   const int32_t gru_size,
                   const bool normalise)
        : m_num_features(num_features),
          m_num_classes(num_classes),
          m_gru_size(gru_size),
          m_normalise(normalise),
          m_gru(torch::nn::GRUOptions(m_num_features, gru_size)
                        .num_layers(2)
                        .bidirectional(true)
                        .batch_first(true)),
          m_linear(2 * m_gru_size, m_num_classes) {
    register_module("gru", m_gru);
    register_module("linear", m_linear);
}

torch::Tensor GRUModel::forward(torch::Tensor x) {
    x = std::move(std::get<0>(m_gru->forward(x)));
    x = m_linear->forward(x);
    if (m_normalise) {
        x = torch::softmax(x, -1);
    }
    return x;
}

}  // namespace dorado::polisher
