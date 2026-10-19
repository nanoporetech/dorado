#pragma once

#include <cstdint>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

bool mps_flstm_is_broken();
int mps_max_safe_batch_size(float memory_limit_fraction,
                            const config::BasecallModelConfig& model_config);
int mps_batch_size_granularity(const config::BasecallModelConfig& model_config);

int metal_lstm_max_safe_batch_size(float memory_limit_fraction,
                                   const config::BasecallModelConfig& model_config);
int metal_lstm_batch_size_granularity();

uint64_t metal_memory_limit(float memory_limit_fraction);

}  // namespace dorado::basecall
