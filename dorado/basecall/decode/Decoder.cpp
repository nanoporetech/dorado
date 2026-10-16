#include "Decoder.h"

#include "CPUDecoder.h"
#include "config/BasecallModelConfig.h"

#if DORADO_CUDA_BUILD
#include "CUDADecoder.h"
#endif

namespace dorado::basecall::decode {

std::size_t DecoderConfig::num_states() const { return std::pow(num_bases, state_len); }

std::size_t DecoderConfig::num_transitions() const { return num_states() * num_bases; }

DecoderOptions create_decoder_options(const config::BasecallModelConfig& config) {
    DecoderOptions options;
    options.q_shift = config.qbias;
    options.q_scale = config.qscale;
    return options;
}

DecoderConfig create_decoder_config(const config::BasecallModelConfig& config) {
    return {create_decoder_options(config), config.is_tx_model() ? config.tx->crf.n_base : 4,
            config.state_len, config.clamp ? 5.f : ((config.scale != 1.f) ? config.scale : 0.f)};
}

std::unique_ptr<Decoder> create_decoder(
        c10::Device device,
        [[maybe_unused]] const config::BasecallModelConfig& config,
        [[maybe_unused]] const std::span<const config::ModBaseModelConfig> modbase_configs) {
#if DORADO_CUDA_BUILD
    if (device.is_cuda()) {
        return std::make_unique<decode::CUDADecoder>(create_decoder_config(config).score_clamp,
                                                     modbase_configs);
    }
#endif
    if (device.is_cpu()) {
        return std::make_unique<decode::CPUDecoder>();
    }

    throw std::runtime_error("Unsupported device type for decoder creation: " + device.str());
}

}  // namespace dorado::basecall::decode
