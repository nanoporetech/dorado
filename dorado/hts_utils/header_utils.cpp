#include "hts_utils/header_utils.h"

#include "utils/string_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dorado {

namespace utils {

namespace {
static constexpr std::pair<std::string_view, HeaderLineType> header_type_lut[]{
        {"HD", HeaderLineType::HD}, {"SQ", HeaderLineType::SQ}, {"RG", HeaderLineType::RG},
        {"PG", HeaderLineType::PG}, {"CO", HeaderLineType::CO},
};
}  // namespace

HeaderLineType parse_header_line_type(std::string_view header_line) {
    if (header_line.size() < 3) {
        return HeaderLineType::UNKNOWN;
    }
    const auto tag = header_line.substr(1, 2);
    for (const auto& ht : header_type_lut) {
        if (ht.first == tag) {
            return ht.second;
        }
    }
    return HeaderLineType::UNKNOWN;
}

std::vector<utils::HeaderLineData> parse_header(
        sam_hdr_t& header,
        const std::set<HeaderLineType>& selected_line_types) {
    const char* hdr_text = sam_hdr_str(&header);
    if (!hdr_text) {
        spdlog::warn("Could not retrieve BAM header text!");
        return {};
    }
    return parse_header(hdr_text, selected_line_types);
}

std::vector<utils::HeaderLineData> parse_header(
        const char* header_text,
        const std::set<HeaderLineType>& selected_line_types) {
    std::vector<utils::HeaderLineData> ret;
    std::istringstream hdr_stream(header_text);
    std::string line;

    while (std::getline(hdr_stream, line)) {
        if (std::size(line) < 3) {
            // Header tag cannot fit in less than 3 chars.
            continue;
        }
        if (line[0] != '@') {
            // Non-header line found.
            break;
        }

        // Split line by tabs.
        const std::vector<std::string_view> fields = utils::split_view(line, '\t');

        // Malformed lines.
        if (std::empty(fields)) {
            continue;
        }

        const HeaderLineType header_type = parse_header_line_type(std::string{fields.front()});
        if (!selected_line_types.empty() && !selected_line_types.contains(header_type)) {
            continue;
        }

        // Split tags by colon.
        std::vector<std::pair<std::string, std::string>> key_value_pairs;
        for (size_t i = 1; i < std::size(fields); ++i) {
            const auto pos = fields[i].find(':');
            if (pos != std::string::npos) {
                std::string key{fields[i].substr(0, pos)};
                std::string value{fields[i].substr(pos + 1)};
                key_value_pairs.emplace_back(std::move(key), std::move(value));
            } else {
                key_value_pairs.emplace_back(fields[i], std::string());
            }
        }

        // Store in the map.
        ret.emplace_back(utils::HeaderLineData{header_type, std::move(key_value_pairs)});
    }

    return ret;
}

void header_to_stream(std::ostream& os, const std::vector<utils::HeaderLineData>& header) {
    for (const auto& line : header) {
        os << fmt::underlying(line.header_type);
        for (const auto& [key, value] : line.tags) {
            os << '\t' << key << ":" << value;
        }
        os << '\n';
    }
}

std::string header_to_string(const std::vector<utils::HeaderLineData>& header) {
    std::ostringstream oss;
    header_to_stream(oss, header);
    return oss.str();
}

}  // namespace utils
}  // namespace dorado
