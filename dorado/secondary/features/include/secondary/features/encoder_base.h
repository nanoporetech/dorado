#pragma once

#include "secondary/common/bam_file.h"
#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/sample.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dorado::secondary {

enum class NormaliseType {
    TOTAL,
    FWD_REV,
};

struct KeyHash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2>& key) const {
        return std::hash<T1>()(key.first) ^ std::hash<T2>()(key.second);
    }
};

using FeatureIndicesType =
        std::unordered_map<std::pair<std::string, bool>, std::vector<int64_t>, KeyHash>;

constexpr auto FeatureTensorType = torch::kFloat32;

enum class FeatureColumns : int32_t {
    BASE,
    QUAL,
    STRAND,
    MAPQ,
    DWELL,
    HAPLOTAG,
    DTYPE,
    SNP_QV,
};

using FeatureColumnMap = std::unordered_map<FeatureColumns, int32_t>;

inline NormaliseType parse_normalise_type(std::string type) {
    // Convert to lower case.
    std::transform(std::begin(type), std::end(type), std::begin(type),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (type == "total") {
        return NormaliseType::TOTAL;
    } else if (type == "fwd_rev") {
        return NormaliseType::FWD_REV;
    }
    throw std::runtime_error{"Unknown normalise type: '" + type + "'!"};
}

class EncoderBase {
public:
    virtual ~EncoderBase() = default;

    virtual secondary::Sample encode_region(const std::string& ref_name,
                                            const int64_t ref_start,
                                            const int64_t ref_end,
                                            const int32_t seq_id) = 0;

    virtual at::Tensor collate(std::vector<at::Tensor> batch) const = 0;

    virtual std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const = 0;

    virtual FeatureColumnMap get_feature_column_map() const = 0;
};

inline std::string feature_column_to_string(const FeatureColumns feature) {
    switch (feature) {
    case FeatureColumns::BASE:
        return "BASE";
    case FeatureColumns::QUAL:
        return "QUAL";
    case FeatureColumns::STRAND:
        return "STRAND";
    case FeatureColumns::MAPQ:
        return "MAPQ";
    case FeatureColumns::DWELL:
        return "DWELL";
    case FeatureColumns::HAPLOTAG:
        return "HAPLOTAG";
    case FeatureColumns::SNP_QV:
        return "SNP_QV";
    case FeatureColumns::DTYPE:
        return "DTYPE";
    default:
        return "UNKNOWN";
    }
}

/**
 * \brief Helper to get the feature column ID.
 * \param feature_column_map The lookup where to find the column.
 * \param feature The name of the feature to find.
 * \return Numeric ID of the feature.
 * \throws If the feature is not found.
 */
inline int32_t get_feature_column_or_throw(const FeatureColumnMap& feature_column_map,
                                           const FeatureColumns feature) {
    const auto it = feature_column_map.find(feature);
    if (it == std::cend(feature_column_map)) {
        throw std::runtime_error{"Cannot find the " + feature_column_to_string(feature) +
                                 " column in the feature_column_map!"};
    }
    return it->second;
}

}  // namespace dorado::secondary
