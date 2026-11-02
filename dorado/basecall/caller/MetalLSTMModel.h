#pragma once

#include "MetalModel.h"
#include "config/BasecallModelConfig.h"

namespace dorado::basecall {

namespace model {
struct MetalCRFModelImpl;
}  // namespace model

class MetalLSTMModel final : public MetalModel {
public:
    MetalLSTMModel(const config::BasecallModelConfig &model_config, float memory_limit_fraction);
    ~MetalLSTMModel() override;

    at::Tensor create_input_tensor() const override;

private:
    MetalLSTMModel(MetalLSTMModel &&other) = delete;
    MetalLSTMModel &operator=(MetalLSTMModel &&) = delete;
    MetalLSTMModel(const MetalLSTMModel &) = delete;
    MetalLSTMModel &operator=(const MetalLSTMModel &) = delete;

private:
    void set_chunk_batch_size(const config::BasecallModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              int chunk_size,
                              int batch_size);
    int benchmark_batch_sizes(const config::BasecallModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              float memory_limit_fraction);
    bool call_task(MetalCallerTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    std::unique_ptr<model::MetalCRFModelImpl> m_model;

    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
};

}  // namespace dorado::basecall
