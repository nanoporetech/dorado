#pragma once

#include "types.h"

namespace dorado {
struct CorrectionAlignments;
}

namespace dorado::correction {

bool extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                     const CorrectionAlignments& alignments,
                     int window_size);

}  // namespace dorado::correction
