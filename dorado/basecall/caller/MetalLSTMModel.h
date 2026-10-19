#pragma once

#include "MetalModel.h"
#include "config/BasecallModelConfig.h"

namespace dorado::basecall {

class MetalDecoder;

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
    DecodedData decode(int chunk_idx) const override;
    bool call_task(MetalCallerTask &task, std::mutex &inter_caller_mutex, int try_count) override;
    MTL::SharedEvent *decode_complete_event() const override {
        return m_decode_complete_event.get();
    }

    const config::BasecallModelConfig m_config;
    std::unique_ptr<model::MetalCRFModelImpl> m_model;
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;
};

}  // namespace dorado::basecall
