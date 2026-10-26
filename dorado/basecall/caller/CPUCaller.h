#pragma once

#include "basecall/ModelCaller.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfigFwd.h"
#include "decode/ModBaseDecoder.h"
#include "modbase/nn/ModBaseModel.h"

#include <torch/nn.h>

#include <atomic>
#include <memory>
#include <optional>
#include <span>

namespace dorado::basecall::decode {
class Decoder;
}

namespace dorado::basecall::model {
class Model;
}

namespace dorado::basecall {

class TensorRunnerState;

class CPUCaller final : public ModelCaller {
public:
    CPUCaller(const config::BasecallModelConfig& model_config,
              std::span<const config::ModBaseModelConfig> modbase_configs);
    ~CPUCaller() override;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(TensorRunnerState& state, int num_chunks);
    size_t num_batch_dims() const override { return 1; }
    const config::BasecallModelConfig& config() const override { return m_config; }
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_modbase_info; }
    void terminate() override {}
    void restart() override {}
    std::string get_name() const override { return "CPUCaller"; }
    stats::NamedStats sample_stats() const override;

private:
    const config::BasecallModelConfig m_config;
    std::unique_ptr<decode::Decoder> m_decoder;
    at::TensorOptions m_options;
    decode::DecoderOptions m_decoder_options;
    std::unique_ptr<model::Model> m_model;
    decode::ModBaseDecoder m_modbase_decoder;
    std::optional<modbase::ModBaseLatentSpaceModels> m_modbase_modules;
    std::shared_ptr<ModBaseInfo> m_modbase_info;

    std::atomic<int64_t> m_model_ms{0};
    std::atomic<int64_t> m_decode_ms{0};
};

}  // namespace dorado::basecall
