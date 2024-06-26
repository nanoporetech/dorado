#pragma once

#include <string>
#include <unordered_map>

namespace dorado::demux {

std::unordered_map<std::string, std::string> parse_custom_sequences(
        const std::string& sequences_file);

}  // namespace dorado::demux
