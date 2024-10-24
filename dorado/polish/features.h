#pragma once

#include "polish/medaka_bamiter.h"

#include <torch/torch.h>

#include <string>
#include <string_view>
#include <vector>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

enum class NormaliseType {
    TOTAL,
    FWD_REV,
};

struct CountsResult {
    torch::Tensor feature_matrix;
    torch::Tensor positions;
};

// struct CountsFeatureEncoderResults {
// };

struct KeyHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& key) const {
        return std::hash<T1>()(key.first) ^ std::hash<T2>()(key.second);
    }
};

// struct CountsFeatureEncoderResults;

class CountsFeatureEncoder {
public:
    CountsFeatureEncoder(bam_fset* bam_set);

    CountsFeatureEncoder(bam_fset* bam_set,
                         const NormaliseType normalise_type,
                         const std::vector<std::string>& dtypes,
                         const std::string_view tag_name,
                         const int32_t tag_value,
                         const bool tag_keep_missing,
                         const std::string_view read_group,
                         const int32_t min_mapq,
                         const bool symmetric_indels);

    CountsResult encode_region(const std::string_view region);

private:
    [[maybe_unused]] bam_fset* m_bam_set = nullptr;
    [[maybe_unused]] NormaliseType m_normalise_type{NormaliseType::TOTAL};
    [[maybe_unused]] std::vector<std::string> m_dtypes;
    [[maybe_unused]] std::string m_tag_name;
    [[maybe_unused]] int32_t m_tag_value = 0;
    [[maybe_unused]] bool m_tag_keep_missing = false;
    [[maybe_unused]] std::string m_read_group;
    [[maybe_unused]] int32_t m_min_mapq = 1;
    [[maybe_unused]] bool m_symmetric_indels = false;

    std::unordered_map<std::pair<std::string, bool>, std::vector<size_t>, KeyHash>
            m_feature_indices;
};

// CountsResult counts_feature_encoder(bam_fset* bam_set, const std::string_view region);

}  // namespace dorado::polisher
