#pragma once

#include "secondary/common/batched_data.h"

#include <torch/nn/module.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
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
    virtual at::Tensor predict_on_batch(BatchedData batched_data);

    /**
     * \brief Move an input batch to the model device before the serialized forward path.
     *        This intentionally avoids taking the model mutex so host-to-device copies can
     *        overlap with another worker's compute once the model has been fully initialized.
     */
    virtual BatchedData prepare_batch_input(BatchedData batched_data, bool non_blocking) const;

    /**
     * \brief Run the model on a batch that is already on the correct device / input precision.
     *        The returned tensor stays on the model device.
     */
    virtual at::Tensor predict_on_device_batch(const BatchedData &batched_data);

    /**
     * \brief Approximate memory consumption estimate given an input batch tensor shape.
     *          This is model specific, and facilitates auto batch size computation
     */
    virtual double estimate_batch_memory(const std::vector<int64_t> &batch_tensor_shape) const = 0;

    /**
     * \brief Getter for the set of non-persistent buffers. Libtorch lacks this feature in `register_buffer`.
     */
    std::unordered_set<std::string> get_non_persistent_buffers() const;

    virtual bool requires_ref(void) const;

protected:
    // Hidden tag to enforce construction via the factory function.
    class MustConstructWithFactory {
        explicit MustConstructWithFactory(int) {}
        friend class ModelTorchBase;
    };
    explicit ModelTorchBase(const MustConstructWithFactory &) {}

    ModelTorchBase(const ModelTorchBase &) = delete;
    ModelTorchBase &operator=(const ModelTorchBase &) = delete;
    ModelTorchBase(ModelTorchBase &&) = delete;
    ModelTorchBase &operator=(ModelTorchBase &&) = delete;

    bool m_normalise = true;
    bool m_half_precision = false;
    mutable std::mutex m_mutex_write;
    std::unordered_set<std::string> m_non_persistent_buffers{};

    void add_nonpersistent_buffer(std::string name);
};

}  // namespace dorado::secondary
