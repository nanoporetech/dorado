#pragma once

#include "basecall/MetalCaller.h"

#include <torch/nn/modules/container/any.h>
#include <torch/nn/pimpl.h>

namespace dorado::basecall {

class MPSCaller final : public MetalCaller {
public:
    MPSCaller(const config::BasecallModelConfig &model_config);
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
    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count);
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    torch::nn::ModuleHolder<torch::nn::AnyModule> m_model;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    at::Tensor m_scores_TNC, m_posts_NTC, m_bwd_NTC;

    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_float_cps, m_fwd_scan_add_softmax_float_cps;
};

}  // namespace dorado::basecall
