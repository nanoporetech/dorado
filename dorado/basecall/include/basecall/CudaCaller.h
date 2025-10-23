
#pragma once

#include "DecodedChunk.h"
#include "ModelRunnerBase.h"
#include "config/BasecallModelConfig.h"
#include "nn/AuxiliaryData.h"
#include "nn/KoiUtils.h"
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
#include <tuple>
#include <vector>

namespace dorado::basecall {

namespace decode {
class Decoder;
}  // namespace decode

class CudaCaller {
public:
    CudaCaller(const BasecallerCreationParams &params);

    ~CudaCaller();
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  nn::AuxiliaryData *aux /* = nullptr */);

    void terminate();
    void restart();

    std::tuple<at::Tensor, at::Tensor, at::Tensor> create_input_output_tensor(
            size_t batch_dims_idx) const;
    std::size_t batch_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].N;
    }
    std::size_t chunk_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].T_in;
    }
    bool variable_chunk_sizes() const { return m_variable_chunk_sizes; }
    size_t num_batch_dims() const { return m_batch_dims.size(); };
    c10::Device device() const { return m_options.device(); }
    const config::BasecallModelConfig &config() const { return m_config; }
    bool is_low_latency() const { return m_low_latency; }
    std::string get_name() const { return std::string("CudaCaller_") + m_device; }
    stats::NamedStats sample_stats() const;
    std::pair<int, int> batch_timeouts_ms() const;

private:
    struct GPUTaskQueue;
    GPUTaskQueue &get_task_queue();

    static int get_batch_size_granularity(const config::BasecallModelConfig &model_config) {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return model_config.is_tx_model() ? 32 : 64;
    }

    std::pair<int64_t, int64_t> calculate_memory_requirements() const;
    void determine_batch_dims(const BasecallerCreationParams &params);

    void start_threads();
    void cuda_thread_fn();

    const config::BasecallModelConfig m_config;
    const std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    const at::TensorOptions m_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    std::atomic<bool> m_terminate{false};
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

    bool m_variable_chunk_sizes{false};

    nn::KoiThreads m_thread_pool;
};

}  // namespace dorado::basecall
