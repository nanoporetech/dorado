#pragma once

#include "secondary/bam_file.h"
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

    virtual secondary::Sample encode_region(secondary::BamFile& bam_file,
                                            const std::string& ref_name,
                                            const int64_t ref_start,
                                            const int64_t ref_end,
                                            const int32_t seq_id) const = 0;

    virtual at::Tensor collate(std::vector<at::Tensor> batch) const = 0;

    virtual std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const = 0;
};

}  // namespace dorado::secondary
