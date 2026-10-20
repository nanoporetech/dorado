#include "CpuModel.h"

#include "TensorRunnerState.h"
#include "model/Model.h"
#include "utils/stats.h"

#include <stdexcept>

namespace dorado::basecall {

CpuModel::CpuModel(const config::BasecallModelConfig& model_config,
                   const std::span<const config::ModBaseModelConfig> modbase_configs)
        : m_config(model_config),
          m_decoder(decode::create_decoder(c10::DeviceType::CPU, model_config, {})),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(c10::DeviceType::CPU)),
          m_decoder_options(decode::create_decoder_options(model_config)),
          m_model(model::load_model(model_config, m_options)),
          m_modbase_decoder(modbase_configs) {
    assert(model_config.has_normalised_basecaller_params());
    assert(model_config.basecaller.batch_size() > 0);
    if (!modbase_configs.empty()) {
        if (!model_config.is_flstm_model()) {
            throw std::runtime_error(
                    "Latent-space modbase models require an FLSTM basecalling model");
        }
        m_modbase_modules = modbase::load_modbase_latent_space_model(
                modbase_configs, m_options,
                static_cast<int>(model_config.basecaller.chunk_size() /
                                 model_config.stride_inner()));
        m_modbase_info = modbase::create_modbase_info(modbase_configs);
    }
}

CpuModel::~CpuModel() = default;

std::unique_ptr<RunnerState> CpuModel::create_runner_state(size_t batch_dims_idx) {
    if (batch_dims_idx != 0) {
        throw std::out_of_range("CPU caller only has one batch dimension");
    }
    auto input = at::zeros({m_config.basecaller.batch_size(), m_config.num_features,
                            m_config.basecaller.chunk_size()},
                           m_options);
    return std::make_unique<TensorRunnerState>(std::move(input), TensorLayout::NCT);
}

std::unique_ptr<CpuBatch> CpuModel::prepare_batch(TensorRunnerState& state, int num_chunks) {
    return std::make_unique<CpuBatch>(state, num_chunks);
}

void CpuModel::submit_batch(CpuBatch& batch) {
    stats::Timer timer;
    auto model_output = m_model->forward(batch.state.input().to(m_options.device()), nullptr,
                                         m_modbase_modules.has_value());
    batch.scores_TNC = model_output.scores.transpose(0, 1).contiguous();
    batch.signal_embeddings = std::move(model_output.signal_embeddings);
    m_model_ms += timer.GetElapsedMS();
}

void CpuModel::decode_batch(CpuBatch& batch,
                            size_t /*unit*/,
                            std::vector<decode::DecodedChunk>& out) {
    at::InferenceMode guard;
    stats::Timer timer;
    out = m_decoder->decode(m_decoder->prepare_decode(
            {std::move(batch.scores_TNC), batch.num_chunks, m_decoder_options}));
    if (m_modbase_modules) {
        const std::span<decode::DecodedChunk> chunks(out);
        m_modbase_decoder.decode(modbase::run_modbase_latent_space_models(
                                         *m_modbase_modules, decode::pack_decoded_chunks(chunks),
                                         batch.signal_embeddings.narrow(0, 0, batch.num_chunks)),
                                 chunks);
    }
    m_decode_ms += timer.GetElapsedMS();
}

stats::NamedStats CpuModel::sample_stats() const {
    return {{"model_ms", double(m_model_ms)}, {"decode_ms", double(m_decode_ms)}};
}

}  // namespace dorado::basecall
