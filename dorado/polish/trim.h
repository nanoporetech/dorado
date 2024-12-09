#pragma once

#include "polish/region.h"
#include "polish/sample.h"

#include <cstdint>
#include <iosfwd>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct TrimInfo {
    int64_t start = 0;
    int64_t end = -1;
    bool heuristic = false;
};

/**
 * \brief Finds the trimming coorindates for each sample, so that they can be spliced.
 *          Optionally, trims off everything before/after the specified region.
 */
std::vector<TrimInfo> trim_samples(const std::vector<Sample>& samples, const Region& region);

bool operator==(const TrimInfo& lhs, const TrimInfo& rhs);

std::ostream& operator<<(std::ostream& os, const TrimInfo& rhs);

}  // namespace dorado::polisher
