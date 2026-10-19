#include "basecall/ModelCallerFactory.h"

#include "CPUCaller.h"
#include "ModelCallerSelection.h"

#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "CudaCaller.h"
#include "CudaCallerPlanning.h"
#elif DORADO_METAL_BUILD
#include "MPSCaller.h"
#include "MetalCallerPlanning.h"
#include "MetalLSTMCaller.h"
#endif

namespace dorado::basecall {

namespace {

using CreateCallerFn = std::shared_ptr<ModelCaller> (*)(const BasecallerCreationParams&);
using BatchSizeGranularityFn = int (*)(const config::BasecallModelConfig&);
using MaxSafeBatchSizeFn = int (*)(const std::string&, float, const config::BasecallModelConfig&);

struct CallerImplementation {
    CreateCallerFn create;
    BatchSizeGranularityFn batch_size_granularity;
    MaxSafeBatchSizeFn max_safe_batch_size;
};

std::shared_ptr<ModelCaller> create_cpu_caller(const BasecallerCreationParams& params) {
    return std::make_shared<CPUCaller>(params.model_config);
}

const CallerImplementation cpu_implementation{
        .create = create_cpu_caller,
        .batch_size_granularity = nullptr,
        .max_safe_batch_size = nullptr,
};

#if DORADO_CUDA_BUILD
std::shared_ptr<ModelCaller> create_cuda_caller(const BasecallerCreationParams& params) {
    return std::make_shared<CudaCaller>(params);
}

int cuda_max_safe_batch_size(const std::string& device,
                             float memory_limit_fraction,
                             const config::BasecallModelConfig& model_config) {
    return basecall::cuda_max_safe_batch_size(device, memory_limit_fraction, model_config);
}

const CallerImplementation cuda_implementation{
        .create = create_cuda_caller,
        .batch_size_granularity = cuda_batch_size_granularity,
        .max_safe_batch_size = cuda_max_safe_batch_size,
};
#elif DORADO_METAL_BUILD
std::shared_ptr<ModelCaller> create_mps_caller(const BasecallerCreationParams& params) {
    return std::make_shared<MPSCaller>(params.model_config);
}

int mps_max_safe_batch_size_for_device(const std::string&,
                                       float memory_limit_fraction,
                                       const config::BasecallModelConfig& model_config) {
    return mps_max_safe_batch_size(memory_limit_fraction, model_config);
}

const CallerImplementation mps_implementation{
        .create = create_mps_caller,
        .batch_size_granularity = mps_batch_size_granularity,
        .max_safe_batch_size = mps_max_safe_batch_size_for_device,
};

std::shared_ptr<ModelCaller> create_metal_lstm_caller(const BasecallerCreationParams& params) {
    return std::make_shared<MetalLSTMCaller>(params.model_config, params.memory_limit_fraction);
}

int metal_lstm_batch_size_granularity_for_config(const config::BasecallModelConfig&) {
    return metal_lstm_batch_size_granularity();
}

int metal_lstm_max_safe_batch_size_for_device(const std::string&,
                                              float memory_limit_fraction,
                                              const config::BasecallModelConfig& model_config) {
    return metal_lstm_max_safe_batch_size(memory_limit_fraction, model_config);
}

const CallerImplementation metal_lstm_implementation{
        .create = create_metal_lstm_caller,
        .batch_size_granularity = metal_lstm_batch_size_granularity_for_config,
        .max_safe_batch_size = metal_lstm_max_safe_batch_size_for_device,
};
#endif

const CallerImplementation& select_caller_implementation(
        const std::string& device,
        [[maybe_unused]] const config::BasecallModelConfig& model_config) {
    if (device == "cpu") {
        return cpu_implementation;
    }
#if DORADO_CUDA_BUILD
    return cuda_implementation;
#elif DORADO_METAL_BUILD
    if (device == "metal") {
        switch (select_metal_model_implementation(model_config)) {
        case MetalModelImplementation::mps:
            return mps_implementation;
        case MetalModelImplementation::lstm:
            return metal_lstm_implementation;
        }
    }
#endif
    throw std::runtime_error("Unsupported device: " + device);
}

}  // namespace

std::shared_ptr<ModelCaller> create_model_caller(const BasecallerCreationParams& params) {
    return select_caller_implementation(params.device, params.model_config).create(params);
}

int model_caller_batch_size_granularity(const std::string& device,
                                        const config::BasecallModelConfig& model_config) {
    const auto planning = select_caller_implementation(device, model_config).batch_size_granularity;
    if (planning == nullptr) {
        throw std::runtime_error("Batch size granularity is unavailable for device: " + device);
    }
    return planning(model_config);
}

int model_caller_max_safe_batch_size(const std::string& device,
                                     float memory_limit_fraction,
                                     const config::BasecallModelConfig& model_config) {
    const auto planning = select_caller_implementation(device, model_config).max_safe_batch_size;
    if (planning == nullptr) {
        throw std::runtime_error("Maximum batch size is unavailable for device: " + device);
    }
    return planning(device, memory_limit_fraction, model_config);
}

uint64_t model_caller_memory_limit(const std::string& device, float memory_limit_fraction) {
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        return cuda_memory_limit(device, memory_limit_fraction);
    }
#elif DORADO_METAL_BUILD
    if (device == "metal") {
        return metal_memory_limit(memory_limit_fraction);
    }
#endif
    throw std::runtime_error("Memory limit is unavailable for device: " + device);
}

}  // namespace dorado::basecall
