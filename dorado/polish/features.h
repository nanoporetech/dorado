#pragma once

#include "polish/medaka_bamiter.h"

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

class CountsFeatureEncoder {
public:
    CountsFeatureEncoder(const NormaliseType normalise);
};

void counts_feature_encoder(const bam_fset* bam_set, const std::string_view region);

}  // namespace dorado::polisher
