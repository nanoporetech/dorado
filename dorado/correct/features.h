#pragma once

#include "types.h"

namespace dorado {
struct CorrectionAlignments;
}

namespace dorado::correction {

std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments,
                                             int window_size);

}  // namespace dorado::correction
