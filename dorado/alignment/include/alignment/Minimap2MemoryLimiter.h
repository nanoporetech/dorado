#pragma once

#include "utils/stats.h"

#include <cstddef>

namespace dorado::alignment {

// Set a limit on the amount of memory that minimap2 can use.
//
// This should be called at most once per process, and only during
// initialisation since it's not thread safe.
//
// Note that this only limits concurrent runtime allocations, ie it
// doesn't stop minimap2 from allocating more than this on one thread
// if it's the only thread using minimap2, and loading of an index
// doesn't count towards this limit.
bool install_mm2_limiter_hooks(std::size_t max_memory_limit_GB, std::size_t num_workers);

// Query the current stats of the limiter.
stats::NamedStats mm2_limiter_stats();

}  // namespace dorado::alignment
