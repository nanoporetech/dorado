#pragma once

namespace kadayashi {

double get_timestamp();

double get_peakrss();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): global setting for the executable.
inline bool KDY_VERBOSE;  // verbosity used by CLI

}  // namespace kadayashi
