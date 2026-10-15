#include "Decoder.h"

#include "CPUDecoder.h"
#include "config/BasecallModelConfig.h"

#if DORADO_CUDA_BUILD
#include "CUDADecoder.h"
#endif

namespace dorado::basecall::decode {

DecoderOptions create_decoder_options(const config::BasecallModelConfig& config) {
    DecoderOptions options;
    options.q_shift = config.qbias;
    options.q_scale = config.qscale;
    return options;
}

std::unique_ptr<Decoder> create_decoder(
        c10::Device device,
        [[maybe_unused]] const config::BasecallModelConfig& config,
        [[maybe_unused]] const std::span<const config::ModBaseModelConfig> modbase_configs) {
#if DORADO_CUDA_BUILD
    if (device.is_cuda()) {
        return std::make_unique<decode::CUDADecoder>(
                config.clamp ? 5.f : ((config.scale != 1.f) ? config.scale : 0.f), modbase_configs);
    }
#endif
    if (device.is_cpu()) {
        return std::make_unique<decode::CPUDecoder>();
    }

    throw std::runtime_error("Unsupported device type for decoder creation: " + device.str());
}

}  // namespace dorado::basecall::decode
