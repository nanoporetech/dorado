#pragma once

#include "sample.h"

#include <cstdint>
#include <vector>

namespace dorado::secondary {

std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples,
                                                   const Sample& new_sample);

}  // namespace dorado::secondary
