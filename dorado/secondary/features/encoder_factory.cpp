#include "secondary/features/encoder_factory.h"

#include "encoder_counts.h"
#include "encoder_read_alignment.h"
#include "secondary/architectures/model_config.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

namespace {
std::string get_value(const std::unordered_map<std::string, std::string>& dict,
                      const std::string& key,
                      const bool throw_on_fail) {
    const auto it = dict.find(key);
    if (it == std::cend(dict)) {
        if (!throw_on_fail) {
            return {};
        }
        throw std::runtime_error{"Cannot find key '" + key + "' in kwargs!"};
    }
    if ((std::size(it->second) >= 2) && (it->second.front() == '"') && (it->second.back() == '"')) {
        return it->second.substr(1, std::size(it->second) - 2);
    }
    return it->second;
}

bool get_bool_value(const std::unordered_map<std::string, std::string>& dict,
                    const std::string& key,
                    const bool throw_on_fail) {
    return (get_value(dict, key, throw_on_fail) == "true") ? true : false;
}
}  // namespace

FeatureEncoderType parse_feature_encoder_type(const std::string& type) {
    if (type == "CountsFeatureEncoder") {
        return FeatureEncoderType::COUNTS_FEATURE_ENCODER;
    } else if (type == "ReadAlignmentFeatureEncoder") {
        return FeatureEncoderType::READ_ALIGNMENT_FEATURE_ENCODER;
    }
    throw std::runtime_error{"Unknown feature encoder type: '" + type + "'!"};
}

std::unique_ptr<EncoderBase> encoder_factory(
        const ModelConfig& config,
        const std::filesystem::path& in_ref_fn,
        const std::filesystem::path& in_bam_aln_fn,
        const std::string& read_group,
        const std::string& tag_name,
        const int32_t tag_value,
        const bool clip_to_zero,
        const double min_snp_accuracy,
        const std::optional<bool>& tag_keep_missing_override,
        const std::optional<int32_t>& min_mapq_override,
        const std::optional<HaplotagSource>& hap_source,
        const std::optional<std::filesystem::path>& phasing_bin_fn,
        const KadayashiOptions& kadayashi_opt) {
    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    const auto& kwargs = config.feature_encoder_kwargs;

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        const std::string normalise = get_value(kwargs, "normalise", true);
        const bool tag_keep_missing = (tag_keep_missing_override)
                                              ? *tag_keep_missing_override
                                              : get_bool_value(kwargs, "tag_keep_missing", true);
        const int32_t min_mapq = (min_mapq_override)
                                         ? *min_mapq_override
                                         : std::stoi(get_value(kwargs, "min_mapq", true));
        const bool sym_indels = get_bool_value(kwargs, "sym_indels", true);

        NormaliseType normalise_type = parse_normalise_type(normalise);

        if (phasing_bin_fn) {
            spdlog::warn(
                    "Phasing bin path is provided, but this feature is not supported with the "
                    "counts feature encoder.");
        }

        std::unique_ptr<EncoderCounts> ret = std::make_unique<EncoderCounts>(
                in_bam_aln_fn, normalise_type, config.feature_encoder_dtypes, tag_name, tag_value,
                tag_keep_missing, read_group, min_mapq, sym_indels, clip_to_zero);

        return ret;

    } else if (feature_encoder_type == FeatureEncoderType::READ_ALIGNMENT_FEATURE_ENCODER) {
        const bool tag_keep_missing = (tag_keep_missing_override)
                                              ? *tag_keep_missing_override
                                              : get_bool_value(kwargs, "tag_keep_missing", true);
        const int32_t min_mapq = (min_mapq_override)
                                         ? *min_mapq_override
                                         : std::stoi(get_value(kwargs, "min_mapq", true));
        const int32_t max_reads = std::stoi(get_value(kwargs, "max_reads", true));
        const bool row_per_read = get_bool_value(kwargs, "row_per_read", true);
        const bool include_dwells = get_bool_value(kwargs, "include_dwells", true);
        const bool include_haplotype_column = get_bool_value(kwargs, "include_haplotype", true);
        const bool include_snp_qv_column = get_bool_value(kwargs, "include_snp_qv", false);
        HaplotagSource hap_source_final = hap_source ? *hap_source : HaplotagSource::UNPHASED;

        // Optional. Config version >= 3 feature.
        const bool right_align_insertions = get_bool_value(kwargs, "right_align_insertions", false);

        if ((hap_source_final == HaplotagSource::BIN_FILE) && !phasing_bin_fn) {
            spdlog::warn(
                    "Haplotag source is the input bin file, but no input bin file is provided! "
                    "Continuing without phasing.");
            hap_source_final = HaplotagSource::UNPHASED;
        }

        std::unique_ptr<EncoderReadAlignment> ret = std::make_unique<EncoderReadAlignment>(
                in_ref_fn, in_bam_aln_fn, config.feature_encoder_dtypes, tag_name, tag_value,
                tag_keep_missing, read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read,
                include_dwells, clip_to_zero, right_align_insertions, include_haplotype_column,
                hap_source_final, phasing_bin_fn, include_snp_qv_column, kadayashi_opt);

        return ret;
    }

    throw std::runtime_error{"Unsupported feature encoder type: " + config.feature_encoder_type};
}

FeatureColumnMap feature_column_map_factory(const ModelConfig& config) {
    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        return EncoderCounts::produce_feature_column_map();

    } else if (feature_encoder_type == FeatureEncoderType::READ_ALIGNMENT_FEATURE_ENCODER) {
        const auto& kwargs = config.feature_encoder_kwargs;
        const bool include_dwells = get_bool_value(kwargs, "include_dwells", false);
        const bool include_haplotype_column = get_bool_value(kwargs, "include_haplotype", false);
        const bool include_snp_qv_column = get_bool_value(kwargs, "include_snp_qv", false);
        const int64_t num_dtypes = std::ssize(config.feature_encoder_dtypes) + 1;
        return EncoderReadAlignment::produce_feature_column_map(
                include_dwells, include_haplotype_column, include_snp_qv_column, (num_dtypes > 1));
    }

    throw std::runtime_error{"Unsupported feature encoder type in feature_column_map_factory: " +
                             config.feature_encoder_type};
}

}  // namespace dorado::secondary
