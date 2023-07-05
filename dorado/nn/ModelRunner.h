#pragma once

#include "../decode/Decoder.h"
#include "CRFModel.h"
#include "utils/stats.h"
#include "utils/stitch.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <atomic>
#include <string>

namespace dorado {

class ModelRunnerBase {
public:
    virtual void accept_chunk(int chunk_idx, const torch::Tensor &chunk) = 0;
    virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual size_t model_stride() const = 0;
    virtual size_t chunk_size() const = 0;
    virtual size_t batch_size() const = 0;
    virtual void terminate() = 0;
    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

template <typename T>
class ModelRunner final : public ModelRunnerBase {
public:
    ModelRunner(const CRFModelConfig &model_config,
                const std::string &device,
                int chunk_size,
                int batch_size);
    void accept_chunk(int chunk_idx, const torch::Tensor &chunk) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;
    size_t model_stride() const final { return m_model_stride; }
    size_t chunk_size() const final { return m_input.size(2); }
    size_t batch_size() const final { return m_input.size(0); }
    void terminate() final {}
    std::string get_name() const final { return "ModelRunner"; }
    stats::NamedStats sample_stats() const final;

private:
    torch::Tensor m_input;
    torch::TensorOptions m_options;
    std::unique_ptr<T> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    size_t m_model_stride;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
    std::atomic<int64_t> m_decode_ms = 0;
};

template <typename T>
ModelRunner<T>::ModelRunner(const CRFModelConfig &model_config,
                            const std::string &device,
                            int chunk_size,
                            int batch_size) {
    m_model_stride = static_cast<size_t>(model_config.stride);

    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;
    m_decoder = std::make_unique<T>();

    m_options = torch::TensorOptions().dtype(T::dtype).device(device);
    m_module = load_crf_model(model_config, m_options);

    // adjust chunk size to be a multiple of the stride
    chunk_size -= chunk_size % m_model_stride;

    m_input = torch::zeros({batch_size, model_config.num_features, chunk_size},
                           torch::TensorOptions().dtype(T::dtype).device(torch::kCPU));
}

template <typename T>
std::vector<DecodedChunk> ModelRunner<T>::call_chunks(int num_chunks) {
    torch::InferenceMode guard;
    dorado::stats::Timer timer;
    auto scores = m_module->forward(m_input.to(m_options.device_opt().value()));
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->beam_search(scores, num_chunks, m_decoder_options);
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    ++m_num_batches_called;
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

template <typename T>
void ModelRunner<T>::accept_chunk(int chunk_idx, const torch::Tensor &chunk) {
    m_input.index_put_({chunk_idx, torch::indexing::Ellipsis}, chunk);
}

template <typename T>
stats::NamedStats ModelRunner<T>::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = m_num_batches_called;
    stats["model_ms"] = m_model_ms;
    stats["decode_ms"] = m_decode_ms;
    return stats;
}

}  // namespace dorado
