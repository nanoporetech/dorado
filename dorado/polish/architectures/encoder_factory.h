#pragma once

#include "encoder_counts.h"
#include "encoder_read_alignment.h"
#include "model_config.h"

#include <memory>
#include <string>

namespace dorado::polisher {

enum class FeatureEncoderType {
    COUNTS_FEATURE_ENCODER,
    READ_ALIGNMENT_FEATURE_ENCODER,
};

FeatureEncoderType parse_feature_encoder_type(const std::string& type);

std::unique_ptr<EncoderBase> encoder_factory(const ModelConfig& config,
                                             const int32_t min_mapq_override);

}  // namespace dorado::polisher