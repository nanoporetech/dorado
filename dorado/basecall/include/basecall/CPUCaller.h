#pragma once

#include "ModelCaller.h"
#include "config/BasecallModelConfig.h"

#include <torch/nn.h>

#include <atomic>
#include <memory>

namespace dorado::basecall::decode {
class Decoder;
}

namespace dorado::basecall {

class CPUCaller final : public ModelCaller {
public:
    explicit CPUCaller(const config::BasecallModelConfig& model_config);
    ~CPUCaller() override;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(RunnerContext& context, int num_chunks) override;
    size_t num_batch_dims() const override { return 1; }
    const config::BasecallModelConfig& config() const override { return m_config; }
    void terminate() override {}
    void restart() override {}
    std::string get_name() const override { return "CPUCaller"; }
    stats::NamedStats sample_stats() const override;

private:
    const config::BasecallModelConfig m_config;
    std::unique_ptr<decode::Decoder> m_decoder;
    at::TensorOptions m_options;
    decode::DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};

    std::atomic<int64_t> m_model_ms{0};
    std::atomic<int64_t> m_decode_ms{0};
};

}  // namespace dorado::basecall
