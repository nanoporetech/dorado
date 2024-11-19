#pragma once

#include "counts_feature_encoder.h"
#include "model_config.h"

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

std::unique_ptr<CountsFeatureEncoder> encoder_factory(const ModelConfig& config);

std::unique_ptr<CountsFeatureDecoder> decoder_factory(const ModelConfig& config);

}  // namespace dorado::polisher