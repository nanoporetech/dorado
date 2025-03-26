#pragma once

#include "interval.h"
#include "secondary/region.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

namespace dorado::polisher {

/**
 * \brief Parses a string of form "[1, 17]" into a std::vector.
 */
std::vector<int32_t> parse_int32_vector(const std::string& input);

}  // namespace dorado::polisher
