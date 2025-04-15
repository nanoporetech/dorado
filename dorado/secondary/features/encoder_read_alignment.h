#pragma once

#include "encoder_base.h"
#include "secondary/bam_file.h"
#include "secondary/consensus/sample.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <cstdint>
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
    EncoderReadAlignment() = default;

    EncoderReadAlignment(const std::vector<std::string>& dtypes,
                         const std::string& tag_name,
                         const int32_t tag_value,
                         const bool tag_keep_missing,
                         const std::string& read_group,
                         const int32_t min_mapq,
                         const int32_t max_reads,
                         const bool row_per_read,
                         const bool include_dwells,
                         const bool include_haplotype);

    ~EncoderReadAlignment() = default;

    secondary::Sample encode_region(secondary::BamFile& bam_file,
                                    const std::string& ref_name,
                                    const int64_t ref_start,
                                    const int64_t ref_end,
                                    const int32_t seq_id) const override;

    at::Tensor collate(std::vector<at::Tensor> batch) const override;

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override;

private:
    std::vector<std::string> m_dtypes;
    int32_t m_num_dtypes = 1;
    std::string m_tag_name;
    int32_t m_tag_value = 0;
    bool m_tag_keep_missing = false;
    std::string m_read_group;
    int32_t m_min_mapq = 1;
    int32_t m_max_reads = 100;
    bool m_row_per_read = false;
    bool m_include_dwells = true;
    bool m_include_haplotype = false;
};

}  // namespace dorado::secondary
