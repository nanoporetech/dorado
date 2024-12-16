#include "region.h"

#include <cstddef>
#include <ostream>
#include <sstream>

namespace dorado::polisher {

std::ostream& operator<<(std::ostream& os, const Region& region) {
    os << region.name << ":" << (region.start + 1) << "-" << region.end;
    return os;
}

std::string region_to_string(const Region& region) {
    std::ostringstream oss;
    oss << region;
    return oss.str();
}

Region parse_region_string(const std::string& region) {
    const size_t colon_pos = region.find(':');
    if (colon_pos == std::string::npos) {
        return {region, -1, -1};
    }

    std::string name = region.substr(0, colon_pos);

    if ((colon_pos + 1) == std::size(region)) {
        return {std::move(name), -1, -1};
    }

    size_t dash_pos = region.find('-', colon_pos + 1);
    dash_pos = (dash_pos == std::string::npos) ? std::size(region) : dash_pos;
    const int64_t start =
            ((dash_pos - colon_pos - 1) == 0)
                    ? -1
                    : std::stoll(region.substr(colon_pos + 1, dash_pos - colon_pos - 1)) - 1;
    const int64_t end =
            ((dash_pos + 1) < std::size(region)) ? std::stoll(region.substr(dash_pos + 1)) : -1;

    return Region{std::move(name), start, end};
}

}  // namespace dorado::polisher
