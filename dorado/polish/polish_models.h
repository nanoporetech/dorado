#pragma once

#include <torch/torch.h>

#include <memory>

namespace dorado::polisher {

class TorchModel : public torch::nn::Module {
public:
    virtual ~TorchModel() = default;

    /**
     * \brief This function is virtual and must be overridden by derived classes.
     */
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /**
     * \brief Helper function to get the device where the model is located.
     */
    torch::Device get_device() const;

    /**
     * \brief Convert the model to half precision.
     */
    virtual void to_half();

    /**
     * \brief Predict on a batch with device and precision handling.
     */
    torch::Tensor predict_on_batch(torch::Tensor x);

protected:
    bool m_half_precision = false;
};

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
