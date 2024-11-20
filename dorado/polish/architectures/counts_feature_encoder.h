#pragma once

#include "polish/architectures/base_feature_encoder.h"
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

struct CountsResult {
    torch::Tensor counts;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
};
struct KeyHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& key) const {
        return std::hash<T1>()(key.first) ^ std::hash<T2>()(key.second);
    }
};
using FeatureIndicesType =
        std::unordered_map<std::pair<std::string, bool>, std::vector<int64_t>, KeyHash>;

constexpr auto FeatureTensorType = torch::kFloat32;

enum class LabelSchemeType {
    HAPLOID,
};

class CountsFeatureEncoder : public BaseFeatureEncoder {
public:
    CountsFeatureEncoder(bam_fset* bam_set);

    CountsFeatureEncoder(bam_fset* bam_set, const int32_t min_mapq);

    CountsFeatureEncoder(bam_fset* bam_set,
                         const NormaliseType normalise_type,
                         const std::vector<std::string>& dtypes,
                         const std::string_view tag_name,
                         const int32_t tag_value,
                         const bool tag_keep_missing,
                         const std::string_view read_group,
                         const int32_t min_mapq,
                         const bool symmetric_indels);

    ~CountsFeatureEncoder() = default;

    Sample encode_region(const std::string& ref_name,
                         const int64_t ref_start,
                         const int64_t ref_end,
                         const int32_t seq_id) const override;

private:
    bam_fset* m_bam_set = nullptr;
    NormaliseType m_normalise_type{NormaliseType::TOTAL};
    std::vector<std::string> m_dtypes;
    std::string m_tag_name;
    int32_t m_tag_value = 0;
    bool m_tag_keep_missing = false;
    std::string m_read_group;
    int32_t m_min_mapq = 1;
    bool m_symmetric_indels = false;

    FeatureIndicesType m_feature_indices;
};

class CountsFeatureDecoder : public BaseFeatureDecoder {
public:
    CountsFeatureDecoder(const LabelSchemeType label_scheme_type);

    ~CountsFeatureDecoder() = default;

    std::vector<ConsensusResult> decode_bases(const torch::Tensor& logits) const override;

private:
    [[maybe_unused]] const LabelSchemeType m_label_scheme_type = LabelSchemeType::HAPLOID;
    std::string m_label_scheme{"*ACGT"};
};

}  // namespace dorado::polisher