#pragma once

#include "CudaCallerPlanning.h"
#include "basecall/DecodedChunk.h"
#include "basecall/ModelCaller.h"
#include "basecall/ModelCallerFactory.h"
#include "config/BasecallModelConfig.h"
#include "modbase/nn/ModBaseModel.h"
#include "nn/AuxiliaryData.h"
#include "nn/KoiThreads.h"
#include "utils/stats.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/nn.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace dorado::basecall {

class CudaRunnerState;

namespace decode {
class Decoder;
}  // namespace decode

namespace model {
class Model;
}

class CudaCaller final : public ModelCaller {
public:
    CudaCaller(const BasecallerCreationParams &params);

    ~CudaCaller() override;
    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(CudaRunnerState &context, int num_chunks);
    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  nn::AuxiliaryData *aux /* = nullptr */);

    void terminate() override;
    void restart() override;

    std::tuple<at::Tensor, at::Tensor, at::Tensor> create_input_output_tensor(
            size_t batch_dims_idx) const;
    std::size_t batch_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].N;
    }
    std::size_t chunk_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].T_in;
    }
    bool variable_chunk_sizes() const { return m_variable_chunk_sizes; }
    size_t num_batch_dims() const override { return m_batch_dims.size(); };
    c10::Device device() const { return m_options.device(); }
    const config::BasecallModelConfig &config() const override { return m_config; }
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_modbase_info; }
    bool is_low_latency() const override { return m_low_latency; }
    std::string get_name() const override { return std::string("CudaCaller_") + m_device; }
    stats::NamedStats sample_stats() const override;
    std::pair<int, int> batch_timeouts_ms() const override;

private:
    struct GPUTaskQueue;
    GPUTaskQueue &get_task_queue();

    void determine_batch_dims(const BasecallerCreationParams &params, bool variable_chunk_sizes);

    void start_threads();
    void cuda_thread_fn();

    const config::BasecallModelConfig m_config;
    const std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    const at::TensorOptions m_options;
    std::unique_ptr<model::Model> m_model;
    std::optional<modbase::ModBaseLatentSpaceModels> m_modbase_modules;
    std::shared_ptr<ModBaseInfo> m_modbase_info;
    bool m_terminate{false};
    std::thread m_cuda_thread;
    int m_num_input_features;
    const bool m_low_latency;
    const PipelineType m_pipeline_type;
    c10::cuda::CUDAStream m_stream;

    // A CudaCaller may accept chunks of multiple different sizes. Smaller sizes will be used to
    // speed up processing of reads that are shorter than the longest chunk size.
    std::vector<CudaBatchDims> m_batch_dims;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called{0};
    std::atomic<int64_t> m_model_decode_ms{0};

    const bool m_variable_chunk_sizes;

    nn::KoiThreads m_thread_pool;
};

}  // namespace dorado::basecall
