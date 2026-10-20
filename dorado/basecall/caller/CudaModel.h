#pragma once

#include "CudaCallerPlanning.h"
#include "basecall/DecodedChunk.h"
#include "basecall/ModelCallerFactory.h"
#include "config/BasecallModelConfig.h"
#include "decode/Decoder.h"
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
#include <tuple>
#include <vector>

namespace dorado::basecall {

namespace model {
class Model;
}

// Owns everything backend-specific about running a basecalling model on a CUDA device: model and
// decoder loading, batch dimension selection, warm-up, tensor layout, and the forward/decode steps.
class CudaModel final {
public:
    // A batch that has been copied to the device and is ready to be run.
    struct Batch {
        Batch(at::Tensor input_, int num_chunks_, nn::AuxiliaryData *aux_)
                : input(std::move(input_)), num_chunks(num_chunks_), aux(aux_) {}
        at::Tensor input;
        int num_chunks;
        nn::AuxiliaryData *aux;
        decode::DecoderResult out;
    };

    explicit CudaModel(const BasecallerCreationParams &params);
    ~CudaModel();

    // Copies `input` to the device and creates auxiliary data if required. Runs on the calling
    // thread, using its current stream.
    Batch prepare_batch(const at::Tensor &input, int num_chunks, nn::AuxiliaryData *aux);

    // Runs the model and the GPU part of decoding. Must be called with a stream guard for stream()
    // active.
    void run_batch(Batch &batch);

    // Runs the CPU part of decoding for a batch that has been run.
    std::vector<decode::DecodedChunk> decode_batch(Batch &batch, at::Tensor &output);

    std::tuple<at::Tensor, at::Tensor, at::Tensor> create_input_output_tensor(
            size_t batch_dims_idx) const;

    std::size_t batch_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].N;
    }
    std::size_t chunk_size(const std::size_t batch_dims_idx) const {
        return m_batch_dims[batch_dims_idx].T_in;
    }
    size_t num_batch_dims() const { return m_batch_dims.size(); }
    bool variable_chunk_sizes() const { return m_variable_chunk_sizes; }
    bool is_low_latency() const { return m_low_latency; }
    c10::Device device() const { return m_options.device(); }
    const std::string &device_name() const { return m_device; }
    const config::BasecallModelConfig &config() const { return m_config; }
    std::shared_ptr<ModBaseInfo> modbase_info() const { return m_modbase_info; }
    const c10::cuda::CUDAStream &stream() const { return m_stream; }
    stats::NamedStats sample_stats() const;

private:
    void determine_batch_dims(const BasecallerCreationParams &params);

    const config::BasecallModelConfig m_config;
    const std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    const at::TensorOptions m_options;
    std::unique_ptr<model::Model> m_model;
    std::optional<modbase::ModBaseLatentSpaceModels> m_modbase_modules;
    std::shared_ptr<ModBaseInfo> m_modbase_info;
    int m_num_input_features;
    const bool m_low_latency;
    const PipelineType m_pipeline_type;
    c10::cuda::CUDAStream m_stream;
    const bool m_variable_chunk_sizes;

    // A CudaModel may accept chunks of multiple different sizes. Smaller sizes will be used to
    // speed up processing of reads that are shorter than the longest chunk size.
    std::vector<cuda::BatchDims> m_batch_dims;

    nn::KoiThreads m_thread_pool;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called{0};
    std::atomic<int64_t> m_model_decode_ms{0};
};

// Per-runner buffers and chunk bookkeeping for a CudaModel.
class CudaRunnerState final {
public:
    CudaRunnerState(const CudaModel &model, size_t batch_dims_idx);

    void accept_chunk(int chunk_idx, const at::Tensor &chunk);

    size_t chunk_size() const { return m_chunk_size; }
    size_t batch_size() const { return m_batch_size; }
    bool variable_chunk_sizes() const { return m_variable_chunk_sizes; }

    void reset();

    // Creates the auxiliary data describing the chunks accepted so far, or nullptr for fixed chunk
    // sizes.
    std::unique_ptr<nn::AuxiliaryData> create_auxiliary_data() const;

    at::Tensor m_input;
    at::Tensor m_output;
    at::Tensor m_aux;
    const size_t m_batch_size;
    const size_t m_chunk_size;
    const bool m_variable_chunk_sizes;
    const bool m_is_tx_model;
    const int m_stride;
    const int m_stride_inner;
    const int m_chunk_size_granularity;
    const int m_max_chunk_size;
    std::vector<std::int32_t> m_chunk_sizes;
    size_t m_chunk_offset{0};
    int m_first_conv_padding{0};
    c10::cuda::CUDAStream m_stream;
};

}  // namespace dorado::basecall
