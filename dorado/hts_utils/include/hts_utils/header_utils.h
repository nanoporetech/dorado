#pragma once

#include <set>
#include <string>
#include <vector>

struct sam_hdr_t;

namespace dorado {
namespace utils {

enum HeaderLineType {
    CO,       // One-line text comment
    HD,       // File-level metadata. Optional.
    PG,       // Program
    RG,       // Read group
    SQ,       // Reference sequence dictionary
    UNKNOWN,  // Unknown header type
};

struct HeaderLineData {
    HeaderLineType header_type;
    std::vector<std::pair<std::string, std::string>> tags;
};

HeaderLineType parse_header_line_type(const std::string& header_line);

std::vector<utils::HeaderLineData> parse_header(
        sam_hdr_t& header,
        const std::set<HeaderLineType>& selected_line_types);

std::vector<utils::HeaderLineData> parse_header(
        const char* header_text,
        const std::set<HeaderLineType>& selected_line_types);

void header_to_stream(std::ostream& os, const std::vector<utils::HeaderLineData>& header);
std::string header_to_string(const std::vector<utils::HeaderLineData>& header);

}  // namespace utils
}  // namespace dorado
