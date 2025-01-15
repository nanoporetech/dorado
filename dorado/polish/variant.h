#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::polisher {

struct Variant {
    int32_t seq_id = -1;
    int64_t pos = -1;
    std::string ref;
    std::string alt;
    std::string filter;
    std::unordered_map<std::string, std::string> info;
    float qual = 0.0f;
    std::vector<std::pair<std::string, int32_t>> genotype;
};

std::ostream& operator<<(std::ostream& os, const Variant& v);

}  // namespace dorado::polisher
