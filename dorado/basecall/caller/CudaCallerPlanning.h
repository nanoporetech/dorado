#pragma once

#include "basecall/ModelCallerFactory.h"

#include <c10/core/Device.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace dorado::basecall::cuda {

struct BatchDims {
    int N;
    int T_in;
    int T_out;
};

struct BatchPlan {
    std::vector<BatchDims> batch_dims;
    std::vector<int> max_batch_sizes;
};

struct MemoryRequirements {
    int64_t crfmodel_bytes_per_chunk_timestep;
    int64_t decode_bytes_per_chunk_timestep;
};

int batch_size_granularity(const config::BasecallModelConfig& model_config);
int64_t memory_limit(c10::Device device, float memory_limit_fraction);
MemoryRequirements memory_requirements(const config::BasecallModelConfig& model_config);
BatchPlan calculate_batch_sizes(c10::Device device,
                                float memory_limit_fraction,
                                const config::BasecallModelConfig& model_config,
                                std::optional<PipelineType> pipeline_type,
                                int requested_batch_size,
                                bool variable_chunk_sizes);

int max_safe_batch_size(c10::Device device,
                        float memory_limit_fraction,
                        const config::BasecallModelConfig& model_config);

}  // namespace dorado::basecall::cuda
