#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <tuple>

namespace dorado::polisher {

struct RegionInt {
    int32_t seq_id = -1;
    int64_t start = 0;
    int64_t end = -1;
};

struct Region {
    std::string name;
    int64_t start = 0;
    int64_t end = -1;
};

std::ostream& operator<<(std::ostream& os, const Region& region);

std::string region_to_string(const Region& region);

Region parse_region_string(const std::string& region);

}  // namespace dorado::polisher
