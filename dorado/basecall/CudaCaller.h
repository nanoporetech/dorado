
#pragma once

#include "CRFModelConfig.h"
#include "decode/Decoder.h"
#include "utils/stats.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/nn.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dorado::basecall {

class CudaCaller {
public:
    CudaCaller(const CRFModelConfig &model_config,
               int chunk_size,
               int batch_size,
               const std::string &device,
               float memory_limit_fraction,
               bool exclusive_gpu_access,
               bool low_latency);

    ~CudaCaller();
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  c10::cuda::CUDAStream stream);

    void terminate();
    void restart();

    std::pair<at::Tensor, at::Tensor> create_input_output_tensor(int chunk_size_idx) const;
    int num_chunk_sizes() const { return int(m_in_chunk_sizes.size()); };
    c10::Device device() const { return m_options.device(); }
    const CRFModelConfig &config() const { return m_config; }
    int batch_timeout_ms() const { return m_low_latency ? 100 : 10000; }

    std::string get_name() const { return std::string("CudaCaller_") + m_device; }

    stats::NamedStats sample_stats() const;

private:
    struct NNTask;

    static int get_batch_size_granularity() {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
    }

    std::pair<int64_t, int64_t> calculate_memory_requirements() const;
    void determine_batch_sizes(float memory_limit_fraction, int batch_size, int chunk_size);

    void start_threads();
    void cuda_thread_fn();

    const CRFModelConfig m_config;
    std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    at::TensorOptions m_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    std::atomic<bool> m_terminate{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_cuda_thread;
    int m_num_input_features;
    std::vector<int> m_batch_sizes, m_in_chunk_sizes, m_out_chunk_sizes;
    bool m_exclusive_gpu_access;
    bool m_low_latency;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
    std::atomic<int64_t> m_decode_ms = 0;
};

}  // namespace dorado::basecall
