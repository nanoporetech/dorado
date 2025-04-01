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

struct CountsResult {
    at::Tensor counts;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
};

class EncoderCounts : public EncoderBase {
public:
    EncoderCounts() = default;

    EncoderCounts(const NormaliseType normalise_type,
                  const std::vector<std::string>& dtypes,
                  const std::string& tag_name,
                  const int32_t tag_value,
                  const bool tag_keep_missing,
                  const std::string& read_group,
                  const int32_t min_mapq,
                  const bool symmetric_indels);

    ~EncoderCounts() = default;

    secondary::Sample encode_region(secondary::BamFile& bam_file,
                                    const std::string& ref_name,
                                    const int64_t ref_start,
                                    const int64_t ref_end,
                                    const int32_t seq_id) const override;

    at::Tensor collate(std::vector<at::Tensor> batch) const override;

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override;

private:
    NormaliseType m_normalise_type{NormaliseType::TOTAL};
    std::vector<std::string> m_dtypes;
    int32_t m_num_dtypes = 1;
    std::string m_tag_name;
    int32_t m_tag_value = 0;
    bool m_tag_keep_missing = false;
    std::string m_read_group;
    int32_t m_min_mapq = 1;
    bool m_symmetric_indels = false;
    FeatureIndicesType m_feature_indices;
};

}  // namespace dorado::secondary
