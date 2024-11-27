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
     * \brief Runs the eval() function, but also allows to abstract the functionality
     *          for derived types, in case they utilize composition.
     */
    virtual void set_eval();

    /**
     * \brief Runs the "to()" function but also allows to abstract the functionality
     *          for derived types, in case they utilize composition.
     */
    virtual void to_device(torch::Device device, bool non_blocking = false);

    /**
     * \brief Predict on a batch with device and precision handling.
     */
    torch::Tensor predict_on_batch(torch::Tensor x);

protected:
    bool m_half_precision = false;
};

}  // namespace dorado::polisher
