#include "Decoder.h"

#include "config/BasecallModelConfig.h"

#if DORADO_CUDA_BUILD
#include "CUDADecoder.h"
#endif

#include "CPUDecoder.h"

#include <c10/core/Device.h>

namespace dorado::basecall::decode {

std::unique_ptr<Decoder> create_decoder(c10::Device device,
                                        const config::BasecallModelConfig& config) {
#if DORADO_CUDA_BUILD
    if (device.is_cuda()) {
        return std::make_unique<decode::CUDADecoder>(config.clamp ? 5.f : 0.f);
    }
#else
    (void)config;  // unused in other build types
#endif
    if (device.is_cpu()) {
        return std::make_unique<decode::CPUDecoder>();
    }

    throw std::runtime_error("Unsupported device type for decoder creation: " + device.str());
}

}  // namespace dorado::basecall::decode
