
#pragma once

#include "BasecallerParams.h"
#include "CRFModelConfig.h"
#include "api/caller_creation.h"
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
    CudaCaller(const BasecallerCreationParams &params);

    ~CudaCaller();
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks);

    void terminate();
    void restart();

    // Default value for timeout for incomplete batches. Large value of 30 seconds is
    // found to give good results with mistures of short and long reads.
    static constexpr int DEFAULT_BATCH_TIMEOUT_MS = 30000;

    // Default value for timeout of incomplete batches for low-latency pipelines. The
    // value of 350 ms has been found to give good adaptive-sampling performance on all
    // platforms.
    static constexpr int DEFAULT_LOW_LATENCY_TIMEOUT_MS = 350;

    std::pair<at::Tensor, at::Tensor> create_input_output_tensor(size_t batch_dims_idx) const;
    size_t num_batch_dims() const { return m_batch_dims.size(); };
    c10::Device device() const { return m_options.device(); }
    const CRFModelConfig &config() const { return m_config; }
    bool is_low_latency() const { return m_low_latency; }
    std::string get_name() const { return std::string("CudaCaller_") + m_device; }
    stats::NamedStats sample_stats() const;

    int batch_timeout_ms() const {
        return m_low_latency ? DEFAULT_LOW_LATENCY_TIMEOUT_MS : DEFAULT_BATCH_TIMEOUT_MS;
    }

private:
    struct NNTask;

    static int get_batch_size_granularity(const CRFModelConfig &model_config) {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return model_config.is_tx_model() ? 32 : 64;
    }

    std::pair<int64_t, int64_t> calculate_memory_requirements() const;
    void determine_batch_dims(const BasecallerCreationParams &params,
                              int batch_size,
                              int chunk_size);

    void start_threads();
    void cuda_thread_fn();

    const CRFModelConfig m_config;
    const std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    const at::TensorOptions m_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    std::atomic<bool> m_terminate{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::thread m_cuda_thread;
    int m_num_input_features;
    const bool m_low_latency;
    const PipelineType m_pipeline_type;
    c10::cuda::CUDAStream m_stream;

    // A CudaCaller may accept chunks of multiple different sizes. Smaller sizes will be used to
    // speed up processing of reads that are shorter than the longest chunk size.
    struct BatchDims {
        int N;      // Batch size
        int T_in;   // Chunk size (in)
        int T_out;  // Chunk size (out), after stride
    };
    std::vector<BatchDims> m_batch_dims;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called{0};
    std::atomic<int64_t> m_model_decode_ms{0};
};

}  // namespace dorado::basecall
