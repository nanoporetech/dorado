#pragma once

#include <string>

namespace utils {

// Calculate a mean qscore from a per-base Q string.
float mean_qscore_from_qstring(const std::string& qstring);

}  // namespace utils