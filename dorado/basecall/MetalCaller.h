#pragma once

#include "decode/Decoder.h"
#include "nn/MetalCRFModel.h"
#include "nn/TxModel.h"

#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <torch/types.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalCaller {
protected:
    MetalCaller(const config::BasecallModelConfig &model_config) : m_config(model_config) {}

public:
    virtual ~MetalCaller();

    virtual at::Tensor create_input_tensor() const = 0;
    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks);

    void terminate();
    void restart();

    const config::BasecallModelConfig &config() const { return m_config; }

    struct NNTask;

protected:
    void start_threads();
    void metal_thread_fn();
    void decode_thread_fn();

    virtual DecodedData decode(int chunk_idx) const = 0;
    virtual bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) = 0;

    const config::BasecallModelConfig m_config;

    std::atomic<bool> m_terminate{false};
    std::atomic<bool> m_terminate_decode{false};

    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::thread m_metal_thread;

    std::deque<std::shared_ptr<NNTask>> m_decode_queue;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::thread> m_decode_threads;
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    decode::DecoderOptions m_decoder_options;

    NS::SharedPtr<MTL::Device> m_device;
};

class MetalLSTMCaller : public MetalCaller {
public:
    MetalLSTMCaller(const config::BasecallModelConfig &model_config, float memory_limit_fraction);

    at::Tensor create_input_tensor() const override {
        // Metal convolution kernels operate with channel ordering (N, T, C).  If m_input
        // is to be submitted directly then it must also have this arrangement.
        // Note that this is not the same as other caller implementations, which
        // have T innermost.
        return torch::zeros({m_batch_size, m_in_chunk_size, m_config.num_features}, torch::kF16);
    }

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

    nn::MetalCRFModel m_model{nullptr};
    torch::ScalarType m_scores_dtype = torch::kChar;
    torch::ScalarType m_posts_dtype = torch::kShort;

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

class MetalTxCaller : public MetalCaller {
public:
    MetalTxCaller(const config::BasecallModelConfig &model_config);

    at::Tensor create_input_tensor() const override {
        // NCT
        return torch::zeros({m_batch_size, m_config.num_features, m_in_chunk_size}, torch::kF16);
    }

private:
    void load_tx_model(const config::BasecallModelConfig &model_config);
    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count);
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    nn::TxModel m_model{nullptr};
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;

    torch::ScalarType m_scores_dtype = torch::kHalf;
    torch::ScalarType m_posts_dtype = torch::kFloat32;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    at::Tensor m_scores_TNC, m_posts_NTC, m_bwd_NTC;

    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_float_cps, m_fwd_scan_add_softmax_float_cps;
};

}  // namespace dorado::basecall
