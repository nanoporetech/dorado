#pragma once

#include "CRFModelConfig.h"
#include "decode/Decoder.h"
#include "nn/MetalCRFModel.h"

#include <ATen/core/TensorBody.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dorado::basecall {

class MetalCaller {
public:
    MetalCaller(const CRFModelConfig &model_config, float memory_limit_fraction);
    ~MetalCaller();

    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks);

    void terminate();
    void restart();

    const CRFModelConfig &config() const { return m_config; }
    at::Tensor create_input_tensor() const;

private:
    struct NNTask;

    void set_chunk_batch_size(const CRFModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              int chunk_size,
                              int batch_size);
    int benchmark_batch_sizes(const CRFModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              float memory_limit_fraction);
    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count);

    void start_threads();
    void metal_thread_fn();
    void decode_thread_fn();

    const CRFModelConfig m_config;
    std::atomic<bool> m_terminate{false};
    std::atomic<bool> m_terminate_decode{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::deque<std::shared_ptr<NNTask>> m_decode_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_metal_thread;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::unique_ptr<std::thread>> m_decode_threads;
    decode::DecoderOptions m_decoder_options;
    nn::MetalCRFModel m_model{nullptr};
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_cps, m_fwd_scan_add_softmax_cps;
    // Used to signal completion of an NNTask's decoding.
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;
    std::vector<at::Tensor> m_scores_int8, m_posts_int16, m_bwd;
    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_out_batch_size;
    // v3 and v4 models have different score scaling requirements.
    float m_score_scale{0.0f};
    // Chunk input channel count.
    int m_num_input_features = -1;
};

}  // namespace dorado::basecall
