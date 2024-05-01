#pragma once

#include "types.h"

#include <string>
#include <vector>

namespace dorado::correction {

std::vector<std::string> decode_windows(const std::vector<WindowFeatures>& wfs);

}  // namespace dorado::correction
