#include "basecall/CPUCaller.h"

#include "basecall/crf_utils.h"
#include "caller/TensorRunnerContext.h"
#include "decode/Decoder.h"

#include <stdexcept>

namespace dorado::basecall {

CPUCaller::CPUCaller(const config::BasecallModelConfig& model_config)
        : m_config(model_config),
          m_decoder(decode::create_decoder(c10::DeviceType::CPU, model_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(c10::DeviceType::CPU)),
          m_decoder_options(decode::create_decoder_options(model_config)),
          m_module(load_crf_model(model_config, m_options)) {
    assert(model_config.has_normalised_basecaller_params());
    assert(model_config.basecaller.batch_size() > 0);
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
    return std::make_unique<TensorRunnerContext>(std::move(input), TensorLayout::NCT);
}

std::vector<decode::DecodedChunk> CPUCaller::call_chunks(RunnerContext& generic_context,
                                                         int num_chunks) {
    auto* context = dynamic_cast<TensorRunnerContext*>(&generic_context);
    if (context == nullptr) {
        throw std::invalid_argument("CPU caller received an incompatible runner context");
    }

    at::InferenceMode guard;
    stats::Timer timer;
    auto scores_TNC =
            m_module->forward(context->input().to(m_options.device())).transpose(0, 1)
                              .contiguous();
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->decode(
            m_decoder->prepare_decode({scores_TNC, num_chunks, m_decoder_options}));
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

stats::NamedStats CPUCaller::sample_stats() const {
    return {{"model_ms", double(m_model_ms)}, {"decode_ms", double(m_decode_ms)}};
}

}  // namespace dorado::basecall
