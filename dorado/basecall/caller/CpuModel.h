#pragma once

#include "CallerModel.h"
#include "TensorRunnerState.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfigFwd.h"
#include "decode/Decoder.h"
#include "decode/ModBaseDecoder.h"
#include "modbase/nn/ModBaseModel.h"

#include <torch/nn.h>

#include <atomic>
#include <memory>
#include <optional>
#include <span>

namespace dorado::basecall::model {
class Model;
}

namespace dorado::basecall {

struct CpuBatch final : InFlightBatch {
    CpuBatch(TensorRunnerState& state_, int num_chunks_) : state(state_), num_chunks(num_chunks_) {}
    TensorRunnerState& state;
    int num_chunks;
    at::Tensor scores_TNC;
    at::Tensor signal_embeddings;
};

class CpuModel final : public TypedCallerModel<CpuModel, TensorRunnerState, CpuBatch> {
public:
    CpuModel(const config::BasecallModelConfig& model_config,
             std::span<const config::ModBaseModelConfig> modbase_configs);
    ~CpuModel() override;

    const config::BasecallModelConfig& config() const override { return m_config; }
    std::unique_ptr<RunnerState> create_runner_state(size_t batch_dims_idx) override;
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_modbase_info; }
    std::string name() const override { return "CPUCaller"; }
    stats::NamedStats sample_stats() const override;

private:
    friend class TypedCallerModel<CpuModel, TensorRunnerState, CpuBatch>;
    std::unique_ptr<CpuBatch> prepare_batch(TensorRunnerState& state, int num_chunks);
    void submit_batch(CpuBatch& batch);
    void decode_batch(CpuBatch& batch, size_t unit, std::vector<decode::DecodedChunk>& out);

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
