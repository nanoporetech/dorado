#pragma once

#include "ModelCaller.h"
#include "config/ModBaseModelConfigFwd.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado::basecall {

enum class PipelineType { simplex_low_latency, simplex, duplex };

struct BasecallerCreationParams {
    const config::BasecallModelConfig& model_config;
    const std::span<const config::ModBaseModelConfig>& modbase_configs;
    const std::string& device;
    float memory_limit_fraction;
    PipelineType pipeline_type;
    float batch_size_time_penalty;
    bool variable_chunk_sizes;
};

std::shared_ptr<ModelCaller> create_model_caller(const BasecallerCreationParams& params);

int model_caller_batch_size_granularity(const std::string& device,
                                        const config::BasecallModelConfig& model_config);
int model_caller_max_safe_batch_size(const std::string& device,
                                     float memory_limit_fraction,
                                     const config::BasecallModelConfig& model_config);
uint64_t model_caller_memory_limit(const std::string& device, float memory_limit_fraction);

}  // namespace dorado::basecall
