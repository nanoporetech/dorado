#pragma once

#include <cstdint>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

namespace mps {
bool flstm_is_broken();
int max_safe_batch_size(float memory_limit_fraction,
                        const config::BasecallModelConfig& model_config);
int batch_size_granularity(const config::BasecallModelConfig& model_config);
}  // namespace mps

namespace metal {
int lstm_max_safe_batch_size(float memory_limit_fraction,
                             const config::BasecallModelConfig& model_config);
int lstm_batch_size_granularity();

uint64_t memory_limit(float memory_limit_fraction);
}  // namespace metal
}  // namespace dorado::basecall
