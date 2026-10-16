#include "basecall/ModelCallerFactory.h"

#include "basecall/CPUCaller.h"

#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#include "utils/memory_utils.h"
#endif

namespace dorado::basecall {

std::shared_ptr<ModelCaller> create_model_caller(const BasecallerCreationParams& params) {
    if (params.device == "cpu") {
        return std::make_shared<CPUCaller>(params.model_config);
    }
#if DORADO_CUDA_BUILD
    return std::make_shared<CudaCaller>(params);
#elif DORADO_METAL_BUILD
    if (params.device == "metal") {
        return MetalCaller::create(params.model_config, params.memory_limit_fraction);
    }
#endif
    throw std::runtime_error("Unsupported device: " + params.device);
}

int model_caller_batch_size_granularity(const std::string& device,
                                        const config::BasecallModelConfig& model_config) {
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        return CudaCaller::get_batch_size_granularity(model_config);
    }
#elif DORADO_METAL_BUILD
    if (device == "metal") {
        return MetalCaller::get_batch_size_granularity(model_config);
    }
#endif
    throw std::runtime_error("Batch size granularity is unavailable for device: " + device);
}

int model_caller_max_safe_batch_size(const std::string& device,
                                     float memory_limit_fraction,
                                     const config::BasecallModelConfig& model_config) {
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        return CudaCaller::get_max_safe_batch_size(device, memory_limit_fraction, model_config);
    }
#elif DORADO_METAL_BUILD
    if (device == "metal") {
        return MetalCaller::get_max_safe_batch_size(memory_limit_fraction, model_config);
    }
#endif
    throw std::runtime_error("Maximum batch size is unavailable for device: " + device);
}

uint64_t model_caller_memory_limit(const std::string& device, float memory_limit_fraction) {
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        return CudaCaller::get_gpu_mem_limit(device, memory_limit_fraction);
    }
#elif DORADO_METAL_BUILD
    if (device == "metal") {
        // This assumes that Dorado is the only significant memory user on the machine.
        return utils::get_apple_physical_memory_bytes() * memory_limit_fraction;
    }
#endif
    throw std::runtime_error("Memory limit is unavailable for device: " + device);
}

}  // namespace dorado::basecall
