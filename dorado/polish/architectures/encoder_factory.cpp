#include "encoder_factory.h"

namespace dorado::polisher {

LabelSchemeType parse_label_scheme_type(const std::string& type) {
    if (type == "HaploidLabelScheme") {
        return LabelSchemeType::HAPLOID;
    }
    throw std::runtime_error{"Unknown label scheme type: '" + type + "'!"};
}

FeatureEncoderType parse_feature_encoder_type(const std::string& type) {
    if (type == "CountsFeatureEncoder") {
        return FeatureEncoderType::COUNTS_FEATURE_ENCODER;
    }
    throw std::runtime_error{"Unknown feature encoder type: '" + type + "'!"};
}

std::unique_ptr<BaseFeatureEncoder> encoder_factory(const ModelConfig& config) {
    const auto get_value = [](const std::unordered_map<std::string, std::string>& dict,
                              const std::string& key) -> std::string {
        const auto it = dict.find(key);
        if (it == std::cend(dict)) {
            throw std::runtime_error{"Cannot find key '" + key + "' in kwargs!"};
        }
        if ((std::size(it->second) >= 2) && (it->second.front() == '"') &&
            (it->second.back() == '"')) {
            return it->second.substr(1, std::size(it->second) - 2);
        }
        return it->second;
    };

    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        const std::string normalise = get_value(config.feature_encoder_kwargs, "normalise");
        const bool tag_keep_missing =
                (get_value(config.feature_encoder_kwargs, "tag_keep_missing") == "true") ? true
                                                                                         : false;
        const int32_t min_mapq = std::stoi(get_value(config.feature_encoder_kwargs, "min_mapq"));
        const bool sym_indels =
                (get_value(config.feature_encoder_kwargs, "sym_indels") == "true") ? true : false;

        NormaliseType normalise_type = parse_normalise_type(normalise);
        const std::string tag_name;
        constexpr int32_t TAG_VALUE = 0;
        const std::string read_group;

        std::unique_ptr<CountsFeatureEncoder> ret = std::make_unique<CountsFeatureEncoder>(
                normalise_type, config.feature_encoder_dtypes, tag_name, TAG_VALUE,
                tag_keep_missing, read_group, min_mapq, sym_indels);

        return ret;
    }

    throw std::runtime_error{"Unsupported feature encoder type: " + config.feature_encoder_type};
}

std::unique_ptr<BaseFeatureDecoder> decoder_factory([[maybe_unused]] const ModelConfig& config) {
    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    const LabelSchemeType label_scheme_type = parse_label_scheme_type(config.label_scheme_type);

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        return std::make_unique<CountsFeatureDecoder>(label_scheme_type);
    }

    throw std::runtime_error{"No decoder available for the feature encoder type: " +
                             config.feature_encoder_type};
}

}  // namespace dorado::polisher
