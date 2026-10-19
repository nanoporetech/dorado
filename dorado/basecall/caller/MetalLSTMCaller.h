#pragma once

#include "MetalCaller.h"

namespace dorado::basecall {

class MetalDecoder;

namespace model {
struct MetalCRFModelImpl;
}  // namespace model

class MetalLSTMCaller final : public MetalCaller {
public:
    MetalLSTMCaller(const config::BasecallModelConfig &model_config, float memory_limit_fraction);
    ~MetalLSTMCaller();

    at::Tensor create_input_tensor() const override;

private:
    MetalLSTMCaller(MetalLSTMCaller &&other) = delete;
    MetalLSTMCaller &operator=(MetalLSTMCaller &&) = delete;
    MetalLSTMCaller(const MetalLSTMCaller &) = delete;
    MetalLSTMCaller &operator=(const MetalLSTMCaller &) = delete;

private:
    void set_chunk_batch_size(const config::BasecallModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              int chunk_size,
                              int batch_size);
    int benchmark_batch_sizes(const config::BasecallModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              float memory_limit_fraction);
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    std::unique_ptr<model::MetalCRFModelImpl> m_model;

    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;
};

}  // namespace dorado::basecall
