#pragma once

#include "read_pipeline/messages.h"
#include "types.h"

namespace dorado::correction {

void extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                     const CorrectionAlignments& alignments,
                     int window_size);

}  // namespace dorado::correction
