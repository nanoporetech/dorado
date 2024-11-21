#pragma once

#include "polish/architectures/base_feature_encoder.h"
// #include "polish/architectures/counts_feature_encoder.h"
#include "polish/bam_file.h"
#include "polish/consensus_result.h"
#include "polish/medaka_bamiter.h"
#include "polish/sample.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <string>
#include <string_view>
#include <vector>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

class ReadAlignmentFeatureEncoder : public BaseFeatureEncoder {
public:
    ReadAlignmentFeatureEncoder() = default;

    ReadAlignmentFeatureEncoder(const int32_t min_mapq);

    ReadAlignmentFeatureEncoder(const NormaliseType normalise_type,
                                const std::vector<std::string>& dtypes,
                                const std::string_view tag_name,
                                const int32_t tag_value,
                                const bool tag_keep_missing,
                                const std::string_view read_group,
                                const int32_t min_mapq,
                                const int32_t max_reads,
                                const bool row_per_read,
                                const bool include_dwells,
                                const bool include_haplotype);

    ~ReadAlignmentFeatureEncoder() = default;

    Sample encode_region(BamFile& bam_file,
                         const std::string& ref_name,
                         const int64_t ref_start,
                         const int64_t ref_end,
                         const int32_t seq_id) const override;

private:
    NormaliseType m_normalise_type{NormaliseType::TOTAL};
    int32_t m_num_dtypes = 1;
    std::vector<std::string> m_dtypes;
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

}  // namespace dorado::polisher