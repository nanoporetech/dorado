#pragma once

#include <ATen/ATen.h>
#include <torch/nn/module.h>

#include <memory>

namespace dorado::secondary {

class ModelTorchBase : public torch::nn::Module {
public:
    virtual ~ModelTorchBase() = default;

    /**
     * \brief This function is virtual and must be overridden by derived classes.
     */
    virtual at::Tensor forward(at::Tensor x) = 0;

    /**
     * \brief Helper function to get the device where the model is located.
     */
    virtual torch::Device get_device() const;

    /**
     * \brief Convert the model to half precision.
     */
    virtual void to_half();

    /**
     * \brief Changes the state of normalisation.
     */
    virtual void set_normalise(const bool val);

    /**
     * \brief Runs the eval() function, but also allows to abstract the functionality
     *          for derived types, in case they utilize composition.
     */
    virtual void set_eval();

    /**
     * \brief Runs the "to()" function but also allows to abstract the functionality
     *          for derived types, in case they utilize composition.
     */
    virtual void to_device(torch::Device device);

    /**
     * \brief Predict on a batch with device and precision handling.
     */
    virtual at::Tensor predict_on_batch(at::Tensor x);

protected:
    bool m_normalise = true;
    bool m_half_precision = false;
};

}  // namespace dorado::secondary
