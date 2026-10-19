#pragma once

#include "basecall/ModelCallerFactory.h"

#include <c10/core/Device.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace dorado::basecall {

struct CudaBatchDims {
    int N;
    int T_in;
    int T_out;
};

struct CudaBatchPlan {
    std::vector<CudaBatchDims> batch_dims;
    std::vector<int> max_batch_sizes;
};

int cuda_batch_size_granularity(const config::BasecallModelConfig& model_config);
int64_t cuda_memory_limit(c10::Device device, float memory_limit_fraction);
int cuda_max_safe_batch_size(c10::Device device,
                             float memory_limit_fraction,
                             const config::BasecallModelConfig& model_config);
std::pair<int64_t, int64_t> cuda_memory_requirements(
        const config::BasecallModelConfig& model_config);
CudaBatchPlan calculate_cuda_batch_sizes(c10::Device device,
                                         float memory_limit_fraction,
                                         const config::BasecallModelConfig& model_config,
                                         std::optional<PipelineType> pipeline_type,
                                         int requested_batch_size,
                                         bool variable_chunk_sizes);

}  // namespace dorado::basecall
