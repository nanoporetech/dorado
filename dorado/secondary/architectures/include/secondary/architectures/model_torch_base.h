#pragma once

#include <ATen/ATen.h>
#include <torch/nn/module.h>

#include <memory>
#include <vector>

namespace dorado::secondary {

constexpr double MEMORY_ESTIMATE_UPPER_CAP = std::numeric_limits<double>::infinity();

class ModelTorchBase : public torch::nn::Module {
public:
    /**
     * \brief Factory function to construct derived types.
     * This enforces that the way they're constructed is compatible with torch::nn::Module.
     */
    template <typename T, typename... Args>
    static std::shared_ptr<T> make(Args &&...args) {
        MustConstructWithFactory ctor_tag{0};
        return std::make_shared<T>(ctor_tag, std::forward<Args>(args)...);
    }

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
    void set_normalise(const bool val);

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

    /**
     * \brief Approximate memory consumption estimate given an input batch tensor shape.
     *          This is model specific, and facilitates auto batch size computation
     */
    virtual double estimate_batch_memory(const std::vector<int64_t> &batch_tensor_shape) const = 0;

protected:
    // Hidden tag to enforce construction via the factory function.
    class MustConstructWithFactory {
        explicit MustConstructWithFactory(int) {}
        friend class ModelTorchBase;
    };
    explicit ModelTorchBase(const MustConstructWithFactory &) {}

    bool m_normalise = true;
    bool m_half_precision = false;
};

}  // namespace dorado::secondary
