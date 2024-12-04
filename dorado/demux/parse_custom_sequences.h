#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::demux {

struct CustomSequence {
    std::string name;
    std::string sequence;
    std::unordered_map<std::string, std::string> tags;
};

std::vector<CustomSequence> parse_custom_sequences(const std::string& sequences_file);

}  // namespace dorado::demux
