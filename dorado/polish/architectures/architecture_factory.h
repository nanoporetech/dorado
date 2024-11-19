#pragma once

#include "model_config.h"
#include "torch_model_base.h"

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

enum class ModelType {
    GRU,
};

enum class LabelSchemeType {
    HAPLOID,
};

enum class FeatureEncoderType {
    COUNTS_FEATURE_ENCODER,
};

class PolishArchitecture {
public:
    ModelType model_type;
    LabelSchemeType label_scheme_type;
    FeatureEncoderType feature_encoder_type;
    std::vector<std::shared_ptr<TorchModel>> models;
};

ModelType parse_model_type(const std::string& type);

LabelSchemeType parse_label_scheme_type(const std::string& type);

FeatureEncoderType parse_feature_encoder_type(const std::string& type);

PolishArchitecture architecture_factory(const ModelConfig& config);

}  // namespace dorado::polisher