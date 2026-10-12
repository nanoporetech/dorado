#pragma once

#include "ModelCaller.h"
#include "ModelRunnerBase.h"
#include "config/BasecallModelConfig.h"
#include "utils/stats.h"

#include <torch/nn.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

namespace dorado::basecall::decode {
class Decoder;
}  // namespace dorado::basecall::decode

namespace dorado::basecall {

class ModelRunner final : public ModelRunnerBase {
public:
    explicit ModelRunner(const config::BasecallModelConfig &model_config);
    explicit ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx);
    ~ModelRunner();

    void accept_chunk(int chunk_idx, const at::Tensor &chunk) final;
    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) final;
    const config::BasecallModelConfig &config() const final;
    size_t chunk_size() const final;
    size_t batch_size() const final;
    bool variable_chunk_sizes() const final;
    std::pair<int, int> batch_timeouts_ms() const final;
    bool is_low_latency() const final;
    void terminate() final;
    void restart() final;
    std::string get_name() const final;
    stats::NamedStats sample_stats() const final;

private:
    ModelRunner(const ModelRunner &) = delete;
    ModelRunner &operator=(const ModelRunner &) = delete;
    ModelRunner(ModelRunner &&) = delete;
    ModelRunner &operator=(ModelRunner &&) = delete;

private:
    std::optional<config::BasecallModelConfig> m_config;
    std::shared_ptr<ModelCaller> m_caller;
    std::unique_ptr<ModelCaller::RunnerContext> m_runner_context;
    std::unique_ptr<decode::Decoder> m_decoder;
    at::TensorOptions m_options;
    decode::DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    at::Tensor m_input_NCT;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
    std::atomic<int64_t> m_decode_ms = 0;
};

}  // namespace dorado::basecall
