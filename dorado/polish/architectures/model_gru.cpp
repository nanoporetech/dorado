#include "model_gru.h"

namespace dorado::polisher {

GRUModel::GRUModel(const int32_t num_features,
                   const int32_t num_classes,
                   const int32_t gru_size,
                   const int32_t num_layers,
                   const bool bidirectional,
                   const bool normalise)
        : m_num_features(num_features),
          m_num_classes(num_classes),
          m_gru_size(gru_size),
          m_num_layers(num_layers),
          m_bidirectional(bidirectional),
          m_normalise(normalise),
          m_gru(torch::nn::GRUOptions(m_num_features, m_gru_size)
                        .num_layers(m_num_layers)
                        .bidirectional(m_bidirectional)
                        .batch_first(true)),
          m_linear(m_num_layers * m_gru_size, m_num_classes) {
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
