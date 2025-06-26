#include "model_gru.h"

namespace dorado::secondary {

ModelGRU::ModelGRU(const int32_t num_features,
                   const int32_t num_classes,
                   const int32_t gru_size,
                   const int32_t num_layers,
                   const bool bidirectional)
        : m_num_features(num_features),
          m_num_classes(num_classes),
          m_gru_size(gru_size),
          m_num_layers(num_layers),
          m_bidirectional(bidirectional),
          m_gru(torch::nn::GRUOptions(m_num_features, m_gru_size)
                        .num_layers(m_num_layers)
                        .bidirectional(m_bidirectional)
                        .batch_first(true)),
          m_linear(m_num_layers * m_gru_size, m_num_classes) {
    register_module("gru", m_gru);
    register_module("linear", m_linear);
}

torch::Tensor ModelGRU::forward(torch::Tensor x) {
    x = std::get<0>(m_gru->forward(x));
    x = m_linear->forward(x);
    return x;
}

double ModelGRU::estimate_batch_memory(const std::vector<int64_t>& batch_tensor_shape) const {
    if (std::size(batch_tensor_shape) != 3) {
        throw std::runtime_error{
                "Input tensor shape is of wrong dimension! Expected 3 sizes, got " +
                std::to_string(std::size(batch_tensor_shape))};
    }

    // Input tensor shape: [batch_size x num_positions x num_features];
    const int64_t batch_size = batch_tensor_shape[0];
    const int64_t num_positions = batch_tensor_shape[1];

    // IMPORTANT: The following equation was determined as part of the DOR-1293 effort.
    return 1.0424 + (0.0003441 * batch_size) + (0.0000019 * num_positions) +
           (-0.0000026 * std::pow(batch_size, 2)) + (0.0000120 * batch_size * num_positions);
}

}  // namespace dorado::secondary
