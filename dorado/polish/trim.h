#pragma once

#include "polish/sample.h"

#include <tuple>
#include <vector>

namespace dorado::polisher {

std::vector<std::tuple<Sample, bool, bool>> trim_samples(const std::vector<Sample>& samples);

}  // namespace dorado::polisher
