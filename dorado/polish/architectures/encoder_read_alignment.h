#pragma once

#include "polish/architectures/encoder_base.h"
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

struct ReadAlignmentTensors {
    torch::Tensor counts;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    std::vector<std::string> read_ids_left;
    std::vector<std::string> read_ids_right;
};

class ReadAlignmentFeatureEncoder : public BaseFeatureEncoder {
public:
    ReadAlignmentFeatureEncoder() = default;

    ReadAlignmentFeatureEncoder(const std::vector<std::string>& dtypes,
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

    torch::Tensor collate(std::vector<torch::Tensor> batch) const override;

    std::vector<polisher::Sample> merge_adjacent_samples(
            std::vector<Sample> samples) const override;

private:
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