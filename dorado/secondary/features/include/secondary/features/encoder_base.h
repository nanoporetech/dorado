#pragma once

#include "feature_columns.h"
#include "secondary/consensus/sample.h"
#include "secondary/kadayashi/local_haplotagging.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::secondary {

enum class NormaliseType {
    TOTAL,
    FWD_REV,
};

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

    virtual kadayashi::varcall_result_t produce_haplotags(const std::string& ref_name,
                                                          const int64_t ref_start,
                                                          const int64_t ref_end) = 0;

    virtual secondary::Sample encode_region(
            const std::string& ref_name,
            const int64_t ref_start,
            const int64_t ref_end,
            const int32_t seq_id,
            const std::unordered_map<std::string, int32_t>& haplotags) = 0;

    virtual at::Tensor collate(std::vector<at::Tensor> batch, bool pinned_memory) const = 0;

    virtual std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const = 0;

    virtual FeatureColumnMap get_feature_column_map() const = 0;

protected:
    EncoderBase() = default;
    EncoderBase(const EncoderBase&) = delete;
    EncoderBase& operator=(const EncoderBase&) = delete;
    EncoderBase(EncoderBase&&) = delete;
    EncoderBase& operator=(EncoderBase&&) = delete;
};

}  // namespace dorado::secondary
