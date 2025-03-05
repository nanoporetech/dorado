#pragma once

#include "ModelRunnerBase.h"
#include "config/BasecallModelConfig.h"
#include "decode/Decoder.h"
#include "utils/stats.h"

#include <torch/nn.h>

#include <atomic>
#include <string>

namespace dorado::basecall {

class ModelRunner final : public ModelRunnerBase {
public:
    ModelRunner(const config::BasecallModelConfig &model_config, const std::string &device);
    void accept_chunk(int chunk_idx, const at::Tensor &chunk) final;
    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) final;
    const config::BasecallModelConfig &config() const final { return m_config; };
    size_t chunk_size() const final { return m_input_NCT.size(2); }
    size_t batch_size() const final { return m_input_NCT.size(0); }
    void terminate() final {}
    void restart() final {}
    std::string get_name() const final { return "ModelRunner"; }
    stats::NamedStats sample_stats() const final;

private:
    const config::BasecallModelConfig m_config;
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
