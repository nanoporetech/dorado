#pragma once

#include "polish/sample.h"

#include <cstdint>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct TrimInfo {
    int64_t start = 0;
    int64_t end = -1;
    bool is_last_in_contig = false;
    bool heuristic = false;
};

std::vector<TrimInfo> trim_samples(const std::vector<Sample>& samples);

}  // namespace dorado::polisher
