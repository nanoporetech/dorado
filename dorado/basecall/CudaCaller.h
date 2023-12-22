
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
               bool exclusive_gpu_access);

    ~CudaCaller();
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  c10::cuda::CUDAStream stream);

    void terminate();
    void restart();

    at::Tensor create_input_tensor() const;
    at::Tensor create_output_tensor() const;
    const CRFModelConfig &config() const { return m_config; }

    std::string get_name() const { return std::string("CudaCaller_") + m_device; }

    stats::NamedStats sample_stats() const;

private:
    struct NNTask;

    static int get_batch_size_granularity() {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
    }

    int determine_batch_size(const CRFModelConfig &model_config,
                             int chunk_size_in,
                             float memory_limit_fraction,
                             bool run_benchmark);
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
    int m_num_input_features, m_batch_size, m_in_chunk_size, m_out_chunk_size;
    bool m_exclusive_gpu_access;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
    std::atomic<int64_t> m_decode_ms = 0;
};

}  // namespace dorado::basecall
