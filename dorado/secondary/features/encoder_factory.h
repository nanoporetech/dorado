#pragma once

#include "encoder_counts.h"
#include "encoder_read_alignment.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace dorado::secondary {

struct ModelConfig;

enum class FeatureEncoderType {
    COUNTS_FEATURE_ENCODER,
    READ_ALIGNMENT_FEATURE_ENCODER,
};

FeatureEncoderType parse_feature_encoder_type(const std::string& type);

std::unique_ptr<EncoderBase> encoder_factory(const ModelConfig& config,
                                             const std::string& read_group,
                                             const std::string& tag_name,
                                             const int32_t tag_value,
                                             const std::optional<bool>& tag_keep_missing_override,
                                             const std::optional<int32_t>& min_mapq_override);

}  // namespace dorado::secondary
