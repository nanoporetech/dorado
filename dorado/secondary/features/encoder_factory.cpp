#include "encoder_factory.h"

#include "secondary/architectures/model_config.h"

#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

FeatureEncoderType parse_feature_encoder_type(const std::string& type) {
    if (type == "CountsFeatureEncoder") {
        return FeatureEncoderType::COUNTS_FEATURE_ENCODER;
    } else if (type == "ReadAlignmentFeatureEncoder") {
        return FeatureEncoderType::READ_ALIGNMENT_FEATURE_ENCODER;
    }
    throw std::runtime_error{"Unknown feature encoder type: '" + type + "'!"};
}

std::unique_ptr<EncoderBase> encoder_factory(const ModelConfig& config,
                                             const std::string& read_group,
                                             const std::string& tag_name,
                                             const int32_t tag_value,
                                             const std::optional<bool>& tag_keep_missing_override,
                                             const std::optional<int32_t>& min_mapq_override) {
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

    const auto get_bool_value = [&get_value](
                                        const std::unordered_map<std::string, std::string>& dict,
                                        const std::string& key) -> bool {
        return (get_value(dict, key) == "true") ? true : false;
    };

    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    const auto& kwargs = config.feature_encoder_kwargs;

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        const std::string normalise = get_value(kwargs, "normalise");
        const bool tag_keep_missing = (tag_keep_missing_override)
                                              ? *tag_keep_missing_override
                                              : get_bool_value(kwargs, "tag_keep_missing");
        const int32_t min_mapq =
                (min_mapq_override) ? *min_mapq_override : std::stoi(get_value(kwargs, "min_mapq"));
        const bool sym_indels = get_bool_value(kwargs, "sym_indels");

        NormaliseType normalise_type = parse_normalise_type(normalise);

        std::unique_ptr<EncoderCounts> ret = std::make_unique<EncoderCounts>(
                normalise_type, config.feature_encoder_dtypes, tag_name, tag_value,
                tag_keep_missing, read_group, min_mapq, sym_indels);

        return ret;

    } else if (feature_encoder_type == FeatureEncoderType::READ_ALIGNMENT_FEATURE_ENCODER) {
        const bool tag_keep_missing = (tag_keep_missing_override)
                                              ? *tag_keep_missing_override
                                              : get_bool_value(kwargs, "tag_keep_missing");
        const int32_t min_mapq =
                (min_mapq_override) ? *min_mapq_override : std::stoi(get_value(kwargs, "min_mapq"));
        const int32_t max_reads = std::stoi(get_value(kwargs, "max_reads"));
        const bool row_per_read = get_bool_value(kwargs, "row_per_read");
        const bool include_dwells = get_bool_value(kwargs, "include_dwells");
        const bool include_haplotype = get_bool_value(kwargs, "include_haplotype");

        std::unique_ptr<EncoderReadAlignment> ret = std::make_unique<EncoderReadAlignment>(
                config.feature_encoder_dtypes, tag_name, tag_value, tag_keep_missing, read_group,
                min_mapq, max_reads, row_per_read, include_dwells, include_haplotype);

        return ret;
    }

    throw std::runtime_error{"Unsupported feature encoder type: " + config.feature_encoder_type};
}

}  // namespace dorado::secondary
