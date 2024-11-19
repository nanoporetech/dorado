#pragma once

#include "torch_model_base.h"

#include <torch/torch.h>

#include <memory>

namespace dorado::polisher {

class GRUModel : public TorchModel {
public:
    GRUModel(const int32_t num_features,
             const int32_t num_classes,
             const int32_t gru_size,
             const bool normalise = true);

    /**
     * \brief Implementes the forward function for inference.
     */
    torch::Tensor forward(torch::Tensor x) override;

private:
    int32_t m_num_features = 10;
    int32_t m_num_classes = 5;
    int32_t m_gru_size = 128;
    bool m_normalise = true;
    torch::nn::GRU m_gru{nullptr};
    torch::nn::Linear m_linear{nullptr};
};

}  // namespace dorado::polisher
