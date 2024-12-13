#pragma once

#include <cstdint>
#include <string>
#include <tuple>

namespace dorado::polisher {

struct Region {
    int32_t seq_id = -1;
    int64_t start = 0;
    int64_t end = -1;
};

std::tuple<std::string, int64_t, int64_t> parse_region_string(const std::string& region);

}  // namespace dorado::polisher
