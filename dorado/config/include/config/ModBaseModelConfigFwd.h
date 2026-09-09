#pragma once

#include <variant>

namespace dorado::config {

struct ConvOnlyModelConfig;
struct ConvLSTMModelConfig;
struct ConvLSTMV2ModelConfig;
struct ConvLSTMV3ModelConfig;

using ModBaseModelConfig = std::variant<ConvOnlyModelConfig,
                                        ConvLSTMModelConfig,
                                        ConvLSTMV2ModelConfig,
                                        ConvLSTMV3ModelConfig>;

}  // namespace dorado::config
