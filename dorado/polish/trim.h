#pragma once

#include "polish/region.h"
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

/**
 * \brief Finds the trimming coorindates for each sample, so that they can be spliced.
 *          Optionally, trims off everything before/after the specified region.
 */
std::vector<TrimInfo> trim_samples(const std::vector<Sample>& samples, const Region region = {});

}  // namespace dorado::polisher
