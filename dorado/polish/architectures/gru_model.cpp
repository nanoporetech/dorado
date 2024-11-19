#include "gru_model.h"

namespace dorado::polisher {

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
