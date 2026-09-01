#pragma once

#include "basecall/MetalCaller.h"

namespace dorado::basecall {

namespace model {
struct MetalCRFModelImpl;
}  // namespace model

class MetalLSTMCaller final : public MetalCaller {
public:
    MetalLSTMCaller(const config::BasecallModelConfig &model_config, float memory_limit_fraction);
    ~MetalLSTMCaller();

    at::Tensor create_input_tensor() const override;

    static int get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &model_config);
    static int get_batch_size_granularity();

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
    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count);
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    std::unique_ptr<model::MetalCRFModelImpl> m_model;

    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    // Batchsize afer division by the out_split
    int m_out_batch_size;

    // v3 scores come from a tanh activation whose [-1, 1] range is packed into bytes.
    // The linear kernel scales to [-127, 127] byte range, after which beam search
    // rescales to the expected [-5, 5].
    // v4 scores come from a clamped [-5, 5] range that is rescaled by the kernel to
    // fit into bytes.
    // In both cases beam search applies the same 5/127 factor to scores.
    float m_score_scale = static_cast<float>(5.0 / 127.0);

    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    std::vector<at::Tensor> m_scores_TNC, m_posts_NTC, m_bwd_NTC;

    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_cps, m_fwd_scan_add_softmax_cps;
};

}  // namespace dorado::basecall
