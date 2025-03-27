#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

struct Variant {
    int32_t seq_id = -1;
    int64_t pos = -1;
    std::string ref;
    std::vector<std::string> alts;
    std::string filter;
    std::unordered_map<std::string, std::string> info;
    float qual = 0.0f;
    std::vector<std::pair<std::string, int32_t>> genotype;
    int64_t rstart = 0;
    int64_t rend = 0;
};

std::ostream& operator<<(std::ostream& os, const Variant& v);

bool operator==(const Variant& lhs, const Variant& rhs);

}  // namespace dorado::secondary
