#pragma once

#include "read_pipeline/messages.h"
#include "types.h"

namespace dorado::correction {

const int TOP_K = 30;

std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments,
                                             int window_size);

}  // namespace dorado::correction
