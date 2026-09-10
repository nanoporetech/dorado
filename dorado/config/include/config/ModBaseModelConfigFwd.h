#pragma once

#include <variant>

namespace dorado::config {

namespace modbase {
struct ConvOnlyModelConfig;
struct ConvLSTMModelConfig;
struct ConvLSTMV2ModelConfig;
struct ConvLSTMV3ModelConfig;
}  // namespace modbase

using ModBaseModelConfig = std::variant<modbase::ConvOnlyModelConfig,
                                        modbase::ConvLSTMModelConfig,
                                        modbase::ConvLSTMV2ModelConfig,
                                        modbase::ConvLSTMV3ModelConfig>;

}  // namespace dorado::config
