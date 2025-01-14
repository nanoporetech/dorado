#pragma once

#include <cstdint>
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
    std::string qual;
    std::vector<std::pair<std::string, int32_t>> genotype;
};

}  // namespace dorado::polisher
