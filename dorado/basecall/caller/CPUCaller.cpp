#include "CPUCaller.h"

#include "TensorRunnerState.h"
#include "TypedRunnerContext.h"
#include "decode/Decoder.h"
#include "model/Model.h"

#include <stdexcept>

namespace dorado::basecall {

CPUCaller::CPUCaller(const config::BasecallModelConfig& model_config,
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
        m_modbase_modules = modbase::load_modbase_latent_space_model(modbase_configs, m_options);
        m_modbase_info = modbase::create_modbase_info(modbase_configs);
    }
}

CPUCaller::~CPUCaller() = default;

std::unique_ptr<ModelCaller::RunnerContext> CPUCaller::create_runner_context(
        size_t batch_dims_idx) {
    if (batch_dims_idx != 0) {
        throw std::out_of_range("CPU caller only has one batch dimension");
    }
    auto input = at::zeros({m_config.basecaller.batch_size(), m_config.num_features,
                            m_config.basecaller.chunk_size()},
                           m_options);
    return std::make_unique<TypedRunnerContext<CPUCaller, TensorRunnerState>>(
            *this, std::move(input), TensorLayout::NCT);
}

std::vector<decode::DecodedChunk> CPUCaller::call_chunks(TensorRunnerState& state, int num_chunks) {
    at::InferenceMode guard;
    stats::Timer timer;
    auto model_output = m_model->forward(state.input().to(m_options.device()), nullptr,
                                         m_modbase_modules.has_value());
    auto scores_TNC = model_output.scores.transpose(0, 1).contiguous();
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->decode(
            m_decoder->prepare_decode({scores_TNC, num_chunks, m_decoder_options}));
    if (m_modbase_modules) {
        const std::span<decode::DecodedChunk> chunks(decoded_chunks);
        m_modbase_decoder.decode(modbase::run_modbase_latent_space_models(
                                         *m_modbase_modules, decode::pack_decoded_chunks(chunks),
                                         model_output.signal_embeddings.narrow(0, 0, num_chunks)),
                                 chunks);
    }
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

stats::NamedStats CPUCaller::sample_stats() const {
    return {{"model_ms", double(m_model_ms)}, {"decode_ms", double(m_decode_ms)}};
}

}  // namespace dorado::basecall
