#pragma once

#include "hts_utils/FastxRandomReader.h"
#include "secondary/common/bam_file.h"
#include "secondary/consensus/sample.h"
#include "secondary/features/encoder_base.h"
#include "secondary/features/haplotag_source.h"
#include "secondary/features/kadayashi_options.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dorado::secondary {

struct ReadAlignmentTensors {
    at::Tensor counts;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    std::vector<std::string> read_ids_left;
    std::vector<std::string> read_ids_right;
};

class EncoderReadAlignment : public EncoderBase {
public:
    EncoderReadAlignment(const std::filesystem::path& in_ref_fn,
                         const std::filesystem::path& in_bam_aln_fn,
                         const std::vector<std::string>& dtypes,
                         const std::string& tag_name,
                         const int32_t tag_value,
                         const bool tag_keep_missing,
                         const std::string& read_group,
                         const int32_t min_mapq,
                         const int32_t max_reads,
                         const double min_snp_accuracy,
                         const bool row_per_read,
                         const bool include_dwells,
                         const bool clip_to_zero,
                         const bool right_align_insertions,
                         const bool include_haplotype_column,  // Add a column to the tensor.
                         const HaplotagSource hap_source,      // Where the column value comes from.
                         const std::optional<std::filesystem::path>& phasing_bin,
                         const bool include_snp_qv_column,
                         const KadayashiOptions& kadayashi_opt);

    ~EncoderReadAlignment() = default;

    kadayashi::varcall_result_t produce_haplotags(const std::string& ref_name,
                                                  const int64_t ref_start,
                                                  const int64_t ref_end) override;

    secondary::Sample encode_region(
            const std::string& ref_name,
            const int64_t ref_start,
            const int64_t ref_end,
            const int32_t seq_id,
            const std::unordered_map<std::string, int32_t>& haplotags) override;

    at::Tensor collate(std::vector<at::Tensor> batch) const override;

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override;

    FeatureColumnMap get_feature_column_map() const override;

    static FeatureColumnMap produce_feature_column_map(const bool include_dwells,
                                                       const bool include_haplotype_column,
                                                       const bool include_snp_qv_column,
                                                       const bool include_dtypes);

private:
    hts_io::FastxRandomReader m_fastx_reader;
    secondary::BamFile m_bam_file;
    std::vector<std::string> m_dtypes;
    int32_t m_num_dtypes = 1;
    std::string m_tag_name;
    int32_t m_tag_value = 0;
    bool m_tag_keep_missing = false;
    std::string m_read_group;
    int32_t m_min_mapq = 1;
    int32_t m_max_reads = 100;
    double m_min_snp_accuracy = 0.0;
    bool m_row_per_read = false;
    bool m_include_dwells = true;
    bool m_include_haplotype_column = false;
    bool m_include_snp_qv_column = false;
    HaplotagSource m_hap_source;
    bool m_clip_to_zero = false;
    bool m_right_align_insertions = false;
    std::optional<std::filesystem::path> m_phasing_bin;
    KadayashiOptions m_kadayashi_opt;
    std::mutex m_mtx;
    FeatureColumnMap m_feature_column_map{};
};

}  // namespace dorado::secondary
