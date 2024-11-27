#pragma once

#include "counts_feature_encoder.h"
#include "model_config.h"
#include "read_alignment_feature_encoder.h"

#include <memory>
#include <string>

namespace dorado::polisher {

enum class FeatureEncoderType {
    COUNTS_FEATURE_ENCODER,
    READ_ALIGNMENT_FEATURE_ENCODER,
};

FeatureEncoderType parse_feature_encoder_type(const std::string& type);

std::unique_ptr<BaseFeatureEncoder> encoder_factory(const ModelConfig& config);

}  // namespace dorado::polisher