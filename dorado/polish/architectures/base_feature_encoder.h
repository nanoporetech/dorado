#pragma once

#include "polish/bam_file.h"
#include "polish/consensus_result.h"
#include "polish/sample.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace dorado::polisher {

enum class NormaliseType {
    TOTAL,
    FWD_REV,
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

inline NormaliseType parse_normalise_type(std::string type) {
    std::transform(std::begin(type), std::end(type), std::begin(type),
                   [](unsigned char c) { return std::tolower(c); });
    if (type == "total") {
        return NormaliseType::TOTAL;
    } else if (type == "fwd_rev") {
        return NormaliseType::FWD_REV;
    }
    throw std::runtime_error{"Unknown normalise type: '" + type + "'!"};
}

class BaseFeatureEncoder {
public:
    virtual ~BaseFeatureEncoder() = default;

    virtual Sample encode_region(BamFile& bam_file,
                                 const std::string& ref_name,
                                 const int64_t ref_start,
                                 const int64_t ref_end,
                                 const int32_t seq_id) const = 0;

    virtual torch::Tensor collate(std::vector<torch::Tensor> batch) const = 0;

    virtual std::vector<polisher::Sample> merge_adjacent_samples(
            std::vector<Sample> samples) const = 0;
};

}  // namespace dorado::polisher