#pragma once

#include "types.h"

#include <unordered_set>
#include <vector>

namespace dorado {
struct CorrectionAlignments;
}

namespace dorado::correction {

// Filter window features to TOP_K best. Returns collection of useful overlap indices
std::unordered_set<int> filter_features(std::vector<std::vector<OverlapWindow>>& windows,
                                        const CorrectionAlignments& alignments);

std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments);

}  // namespace dorado::correction
