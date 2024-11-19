#pragma once

#include "polish/consensus_result.h"
#include "polish/sample.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <string>
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

    virtual Sample encode_region(const std::string& ref_name,
                                 const int64_t ref_start,
                                 const int64_t ref_end,
                                 const int32_t seq_id) const = 0;
};

class BaseFeatureDecoder {
public:
    virtual ~BaseFeatureDecoder() = default;

    virtual std::vector<ConsensusResult> decode_bases(const torch::Tensor& logits) const = 0;
};

}  // namespace dorado::polisher