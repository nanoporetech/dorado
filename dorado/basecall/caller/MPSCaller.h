#pragma once

#include "basecall/MetalCaller.h"

#include <torch/nn/modules/container/any.h>
#include <torch/nn/pimpl.h>

namespace dorado::basecall {

class MetalDecoder;

class MPSCaller final : public MetalCaller {
public:
    explicit MPSCaller(const config::BasecallModelConfig &model_config);
    ~MPSCaller();

    at::Tensor create_input_tensor() const override;

    static int get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &model_config);
    static int get_batch_size_granularity(const config::BasecallModelConfig &model_config);

private:
    MPSCaller(MPSCaller &&other) = delete;
    MPSCaller &operator=(MPSCaller &&) = delete;
    MPSCaller(const MPSCaller &) = delete;
    MPSCaller &operator=(const MPSCaller &) = delete;

private:
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    torch::nn::ModuleHolder<torch::nn::AnyModule> m_model;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;
};

}  // namespace dorado::basecall
