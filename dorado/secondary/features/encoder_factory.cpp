#include "secondary/features/encoder_factory.h"

#include "encoder_counts.h"
#include "encoder_read_alignment.h"
#include "secondary/architectures/model_config.h"
#include "secondary/architectures/model_config_validation.h"
#include "utils/log_utils.h"

#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

namespace {
inline std::string get_value(const ModelConfig& config, const std::string& key) {
    return get_model_config_feature_encoder_value(config, key);
}

inline bool get_bool_value(const ModelConfig& config, const std::string& key) {
    return get_model_config_feature_encoder_value(config, key) == "true";
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
        const KadayashiOptions& kadayashi_opt,
        const bool legacy_feature_gen) {
    const FeatureEncoderType feature_encoder_type =
            parse_feature_encoder_type(config.feature_encoder_type);

    if (feature_encoder_type == FeatureEncoderType::COUNTS_FEATURE_ENCODER) {
        const std::string normalise = get_value(config, "normalise");
        const bool tag_keep_missing = (tag_keep_missing_override)
                                              ? *tag_keep_missing_override
                                              : get_bool_value(config, "tag_keep_missing");
        const int32_t min_mapq =
                (min_mapq_override) ? *min_mapq_override : std::stoi(get_value(config, "min_mapq"));
        const bool sym_indels = get_bool_value(config, "sym_indels");

        NormaliseType normalise_type = parse_normalise_type(normalise);

        if (phasing_bin_fn) {
            DORADO_LOG(warn,
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
                                              : get_bool_value(config, "tag_keep_missing");
        const int32_t min_mapq =
                (min_mapq_override) ? *min_mapq_override : std::stoi(get_value(config, "min_mapq"));
        const int32_t max_reads = std::stoi(get_value(config, "max_reads"));
        const bool row_per_read = get_bool_value(config, "row_per_read");
        const bool include_dwells = get_bool_value(config, "include_dwells");
        const bool include_haplotype_column = get_bool_value(config, "include_haplotype");
        const bool include_snp_qv_column = get_bool_value(config, "include_snp_qv");
        HaplotagSource hap_source_final = hap_source ? *hap_source : HaplotagSource::UNPHASED;

        // Optional. Config version >= 3 feature.
        const bool right_align_insertions = get_bool_value(config, "right_align_insertions");

        if ((hap_source_final == HaplotagSource::BIN_FILE) && !phasing_bin_fn) {
            DORADO_LOG(warn,
                       "Haplotag source is the input bin file, but no input bin file is provided! "
                       "Continuing without phasing.");
            hap_source_final = HaplotagSource::UNPHASED;
        }

        std::unique_ptr<EncoderReadAlignment> ret = std::make_unique<EncoderReadAlignment>(
                in_ref_fn, in_bam_aln_fn, config.feature_encoder_dtypes, tag_name, tag_value,
                tag_keep_missing, read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read,
                include_dwells, clip_to_zero, right_align_insertions, include_haplotype_column,
                hap_source_final, phasing_bin_fn, include_snp_qv_column, kadayashi_opt,
                legacy_feature_gen);

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
        const bool include_dwells = get_bool_value(config, "include_dwells");
        const bool include_haplotype_column = get_bool_value(config, "include_haplotype");
        const bool include_snp_qv_column = get_bool_value(config, "include_snp_qv");
        const int64_t num_dtypes = std::ssize(config.feature_encoder_dtypes) + 1;
        return EncoderReadAlignment::produce_feature_column_map(
                include_dwells, include_haplotype_column, include_snp_qv_column, (num_dtypes > 1));
    }

    throw std::runtime_error{"Unsupported feature encoder type in feature_column_map_factory: " +
                             config.feature_encoder_type};
}

}  // namespace dorado::secondary
