#include "polish/architectures/read_alignment_feature_encoder.h"

#include "polish/medaka_read_matrix.h"

#include <spdlog/spdlog.h>
#include <utils/timer_high_res.h>

namespace dorado::polisher {

ReadAlignmentFeatureEncoder::ReadAlignmentFeatureEncoder(const int32_t min_mapq)
        : m_min_mapq{min_mapq} {}

ReadAlignmentFeatureEncoder::ReadAlignmentFeatureEncoder(const std::vector<std::string>& dtypes,
                                                         const std::string_view tag_name,
                                                         const int32_t tag_value,
                                                         const bool tag_keep_missing,
                                                         const std::string_view read_group,
                                                         const int32_t min_mapq,
                                                         const int32_t max_reads,
                                                         const bool row_per_read,
                                                         const bool include_dwells,
                                                         const bool include_haplotype)
        : m_num_dtypes{static_cast<int32_t>(std::size(m_dtypes)) + 1},
          m_dtypes{dtypes},
          m_tag_name{tag_name},
          m_tag_value{tag_value},
          m_tag_keep_missing{tag_keep_missing},
          m_read_group{read_group},
          m_min_mapq{min_mapq},
          m_max_reads{max_reads},
          m_row_per_read{row_per_read},
          m_include_dwells{include_dwells},
          m_include_haplotype{include_haplotype} {}

Sample ReadAlignmentFeatureEncoder::encode_region(BamFile& bam_file,
                                                  const std::string& ref_name,
                                                  const int64_t ref_start,
                                                  const int64_t ref_end,
                                                  const int32_t seq_id) const {
    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();

    // Compute the counts and data.
    ReadAlignmentData counts = calculate_read_alignment(
            bam_file, ref_name, ref_start, ref_end, m_num_dtypes, m_dtypes, m_tag_name, m_tag_value,
            m_tag_keep_missing, read_group_ptr, m_min_mapq, m_row_per_read, m_include_dwells,
            m_include_haplotype, m_max_reads);

    (void)counts;
    (void)seq_id;

    return {};

    // // Create Torch tensors from the pileup.
    // const size_t n_rows = std::size(PILEUP_BASES) * m_num_dtypes;
    // CountsResult tensors = read_matrix_data_to_tensors(counts, n_rows);

    // if (!pileup_tensors.counts.numel()) {
    //     const std::string region =
    //             ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);
    //     spdlog::warn("Pileup-feature is zero-length for {} indicating no reads in this region.",
    //                  region);
    //     return {};
    // }

    // return counts_to_features(pileup_tensors, seq_id, m_symmetric_indels, m_feature_indices,
    //                           m_normalise_type);
}

}  // namespace dorado::polisher
