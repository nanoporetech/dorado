#pragma once

#include "model_config.h"
#include "torch_model_base.h"

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

enum class LabelSchemeType {
    HAPLOID,
};

enum class FeatureEncoderType {
    COUNTS_FEATURE_ENCODER,
};

LabelSchemeType parse_label_scheme_type(const std::string& type);

FeatureEncoderType parse_feature_encoder_type(const std::string& type);

}  // namespace dorado::polisher