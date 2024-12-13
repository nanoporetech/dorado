#pragma once

#include "model_torch_base.h"

#include <ATen/ATen.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>

#include <cstdint>
#include <memory>

namespace dorado::polisher {

class ModelGRU : public ModelTorchBase {
public:
    ModelGRU(const int32_t num_features,
             const int32_t num_classes,
             const int32_t gru_size,
             const int32_t num_layers,
             const bool bidirectional,
             const bool normalise);

    at::Tensor forward(at::Tensor x) override;

private:
    int32_t m_num_features = 10;
    int32_t m_num_classes = 5;
    int32_t m_gru_size = 128;
    int32_t m_num_layers = 2;
    bool m_bidirectional = true;
    bool m_normalise = true;
    torch::nn::GRU m_gru{nullptr};
    torch::nn::Linear m_linear{nullptr};
};

}  // namespace dorado::polisher