#include "bam_file.h"

#include <spdlog/spdlog.h>

#include <ostream>
#include <sstream>

BamFile::BamFile(const std::filesystem::path& in_fn)
        : m_fp{hts_open(in_fn.string().c_str(), "rb"), hts_close},
          m_idx{nullptr, hts_idx_destroy},
          m_hdr{nullptr, sam_hdr_destroy},
          m_current_record{bam_init1(), bam_destroy1} {
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    m_idx = std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)>(
            sam_index_load(m_fp.get(), in_fn.string().c_str()), hts_idx_destroy);

    if (!m_idx) {
        throw std::runtime_error{"Could not open index for BAM file: '" + in_fn.string() + "'!"};
    }

    m_hdr = std::unique_ptr<sam_hdr_t, decltype(&sam_hdr_destroy)>(sam_hdr_read(m_fp.get()),
                                                                   sam_hdr_destroy);

    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }

    if (!m_current_record) {
        throw std::runtime_error{"Failed to initialize BAM record"};
    }
}

std::vector<HeaderLineData> BamFile::parse_header() const {
    const char* hdr_text = sam_hdr_str(m_hdr.get());
    if (!hdr_text) {
        spdlog::warn("Could not retrieve BAM header text!");
        return {};
    }

    std::vector<HeaderLineData> ret;
    std::istringstream hdr_stream(hdr_text);
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
        std::vector<std::string> fields;
        std::string field;
        std::istringstream line_stream(line);
        while (std::getline(line_stream, field, '\t')) {
            fields.emplace_back(std::move(field));
        }

        // Malformed lines.
        if (std::empty(fields)) {
            continue;
        }

        // Split tags by colon.
        std::vector<std::pair<std::string, std::string>> key_value_pairs;
        for (size_t i = 1; i < std::size(fields); ++i) {
            const auto pos = fields[i].find(':');
            if (pos != std::string::npos) {
                std::string key = fields[i].substr(0, pos);
                std::string value = fields[i].substr(pos + 1);
                key_value_pairs.emplace_back(std::move(key), std::move(value));
            } else {
                key_value_pairs.emplace_back(fields[i], std::string());
            }
        }

        // Store in the map.
        const std::string& header_type = fields.front();
        ret.emplace_back(HeaderLineData{header_type, std::move(key_value_pairs)});
    }

    return ret;
}

const bam1_t* BamFile::get_next() {
    m_current_record.release();
    m_current_record = std::unique_ptr<bam1_t, decltype(&bam_destroy1)>(bam_init1(), bam_destroy1);

    if (m_current_record == nullptr) {
        return nullptr;
    }

    if (sam_read1(m_fp.get(), m_hdr.get(), m_current_record.get()) >= 0) {
        return m_current_record.get();  // Return the managed BAM record.
    }

    return nullptr;
}

void header_to_stream(std::ostream& os, const std::vector<HeaderLineData>& header) {
    for (const auto& line : header) {
        os << line.header_type;
        for (const auto& [key, value] : line.tags) {
            os << '\t' << key << ":" << value;
        }
        os << '\n';
    }
}

std::string header_to_string(const std::vector<HeaderLineData>& header) {
    std::ostringstream oss;
    header_to_stream(oss, header);
    return oss.str();
}
