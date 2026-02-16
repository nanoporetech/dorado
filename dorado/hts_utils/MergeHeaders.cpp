#include "hts_utils/MergeHeaders.h"

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
void update_and_add_pg_line(sam_hdr_t* hdr, const std::string& key, std::string line) {
    std::string new_id = sam_hdr_pg_id(hdr, key.c_str());
    auto pos = line.find(key);
    line.replace(pos, key.size(), new_id);
    sam_hdr_add_lines(hdr, line.c_str(), 0);
}

std::string kv_to_tag_string(const std::map<std::string, std::string>& additional_tags) {
    std::string result;
    for (const auto& [key, value] : additional_tags) {
        result.append("\t");
        result.append(key);
        result.append(":");
        result.append(value);
    }
    return result;
}

}  // anonymous namespace

namespace dorado::utils {

MergeHeaders::MergeHeaders(bool strip_alignment) : m_strip_alignment(strip_alignment) {}

void MergeHeaders::add_header(sam_hdr_t* hdr, const std::string& filename) {
    return add_header(hdr, filename, "");
}

void MergeHeaders::add_header(sam_hdr_t* hdr,
                              const std::string& filename,
                              const std::string& read_group_selection) {
    // Append the filepath to index into sq_mappings
    m_filepaths.push_back(filename);

    if (!m_strip_alignment) {
        auto res = check_and_add_ref_data(hdr);
        if (res == -1) {
            throw std::runtime_error("Error merging header " + filename +
                                     ". Invalid SQ line in header.");
        }
        if (res == -2) {
            throw std::runtime_error("Error merging header " + filename +
                                     ". SQ lines are incompatible.");
        }
    }

    auto res = check_and_add_rg_data(hdr, read_group_selection);
    if (res == -1) {
        throw std::runtime_error("Error merging header " + filename +
                                 ". Invalid RG line in header.");
    }
    if (res == -2) {
        throw std::runtime_error("Error merging header " + filename +
                                 ". RG lines are incompatible.");
    }

    res = add_pg_data(hdr);
    if (res < 0) {
        throw std::runtime_error("Error merging header " + filename +
                                 ". Invalid PG line in header.");
    }

    add_other_lines(hdr);
}

sam_hdr_t* MergeHeaders::get_merged_header() const {
    if (!m_merged_header) {
        throw std::logic_error(
                "Error in MergeHeaders. get_merged_header() called before finalize_merge().");
    }
    return m_merged_header.get();
}

std::vector<std::vector<uint32_t>> MergeHeaders::get_sq_mapping() const {
    if (!m_merged_header) {
        throw std::logic_error(
                "Error in MergeHeaders. get_sq_mapping() called before finalize_merge().");
    }
    return m_sq_mapping;
}

const std::vector<uint32_t>& MergeHeaders::get_sq_mapping(const std::string& filename) const {
    if (!m_merged_header) {
        throw std::logic_error(
                "Error in MergeHeaders. get_sq_mapping() called before finalize_merge().");
    }

    const auto it = std::find(m_filepaths.cbegin(), m_filepaths.cend(), filename);
    if (it == m_filepaths.cend()) {
        spdlog::error("Couldn't find '{}' in MergedHeaders.", filename);
        throw std::runtime_error("Error in MergeHeaders. get_sq_mapping() couldn't find file.");
    }

    const auto index = std::distance(m_filepaths.cbegin(), it);
    return m_sq_mapping[index];
}

int MergeHeaders::check_and_add_ref_data(sam_hdr_t* hdr) {
    KString tag_wrapper(1000000);
    auto tag_data = tag_wrapper.get();
    auto get_tag_string = [&](const char* tag, int index) -> std::string {
        std::string tag_string;
        if (sam_hdr_find_tag_pos(hdr, "SQ", index, tag, &tag_data) == 0) {
            return std::string(ks_str(&tag_data));
        }
        return {};
    };

    std::map<std::string, RefInfo> ref_map;
    int nrefs = sam_hdr_nref(hdr);
    for (int i = 0; i < nrefs; ++i) {
        std::string ref_name = sam_hdr_line_name(hdr, "SQ", i);
        if (ref_name.empty()) {
            return -1;
        }

        SQLine sq_line{
                .name = std::move(ref_name),
                .reflen = get_tag_string("LN", i),
                .md5 = get_tag_string("M5", i),
                .url = get_tag_string("UR", i),
        };
        const std::string& key = sq_line.name;
        RefInfo info{uint32_t(i), sq_line};
        ref_map[key] = std::move(info);
        auto entry = m_ref_lut.find(key);
        if (entry == m_ref_lut.end()) {
            m_ref_lut[key] = std::move(sq_line);
        } else {
            SQLineComparator cmp;
            if (!cmp(sq_line, entry->second)) {
                return -2;
            }
        }
    }
    m_ref_info_lut.emplace_back(std::move(ref_map));
    return 0;
}

int MergeHeaders::check_and_add_rg_data(sam_hdr_t* hdr, const std::string& read_group_selection) {
    int num_lines = sam_hdr_count_lines(hdr, "RG");
    for (int i = 0; i < num_lines; ++i) {
        auto idp = sam_hdr_line_name(hdr, "RG", i);
        if (!idp) {
            return -1;
        }

        // Filter only useful read groups
        std::string read_group_id(idp);
        if (!read_group_selection.empty() && read_group_selection != read_group_id) {
            continue;
        }

        // Read the RG line
        KString line_wrapper(1000000);
        auto line_data = line_wrapper.get();
        auto res = sam_hdr_find_line_pos(hdr, "RG", i, &line_data);
        if (res < 0) {
            return -1;
        }
        std::string read_group_line(ks_str(&line_data));

        // Add the RG_line to the LUT or error if it a different record already exists
        if (!add_rg(read_group_id, read_group_line)) {
            return -2;
        }
    }
    return 0;
}

int MergeHeaders::add_pg_data(sam_hdr_t* hdr) {
    int num_lines = sam_hdr_count_lines(hdr, "PG");
    for (int i = 0; i < num_lines; ++i) {
        auto idp = sam_hdr_line_name(hdr, "PG", i);
        if (!idp) {
            return -1;
        }
        KString line_wrapper(1000000);
        auto line_data = line_wrapper.get();
        auto res = sam_hdr_find_line_pos(hdr, "PG", i, &line_data);
        if (res < 0) {
            return -1;
        }
        std::string key(idp);
        std::string line(ks_str(&line_data));
        m_program_lut[key].insert(std::move(line));
    }
    return 0;
}

void MergeHeaders::add_other_lines(sam_hdr_t* hdr) {
    auto header_str = sam_hdr_str(hdr);
    if (!header_str) {
        // A nullptr here means the header object is empty. This usually just means
        // the input file was a fastq file.
        return;
    }
    auto source_stream = std::stringstream{sam_hdr_str(hdr)};
    for (std::string header_line; std::getline(source_stream, header_line);) {
        std::string_view header_type = std::string_view(header_line).substr(0, 3);
        if (header_type == "@PG" || header_type == "@RG" || header_type == "@SQ") {
            continue;
        }
        if (header_type == "@HD") {
            // We will generate our own HD line for the output file.
            continue;
        }
        m_other_lines.insert(header_line);
    }
}

void MergeHeaders::finalize_merge() {
    m_merged_header.reset(sam_hdr_init());
    add_hd_header_line(m_merged_header.get());

    for (const auto& entry : m_program_lut) {
        const auto& key = entry.first;
        const auto& lines = entry.second;
        bool first = true;
        for (const auto& line : lines) {
            if (first) {
                sam_hdr_add_lines(m_merged_header.get(), line.c_str(), 0);
                first = false;
            } else {
                update_and_add_pg_line(m_merged_header.get(), key, line);
            }
        }
    }

    for (const auto& entry : m_read_group_lut) {
        const auto& line = entry.second;
        sam_hdr_add_lines(m_merged_header.get(), line.c_str(), 0);
    }

    std::map<std::string, uint32_t> new_sq_order;
    uint32_t sq_idx = 0;
    for (const auto& entry : m_ref_lut) {
        const auto& key = entry.first;
        const auto& sq_line = entry.second;
        sam_hdr_add_line(m_merged_header.get(), "SQ", "SN", sq_line.name.c_str(), "LN",
                         sq_line.reflen.c_str(), !sq_line.md5.empty() ? "M5" : nullptr,
                         !sq_line.md5.empty() ? sq_line.md5.c_str() : nullptr,
                         !sq_line.url.empty() ? "UR" : nullptr,
                         !sq_line.url.empty() ? sq_line.url.c_str() : nullptr, nullptr);
        new_sq_order[key] = sq_idx++;
    }

    for (const auto& line : m_other_lines) {
        sam_hdr_add_lines(m_merged_header.get(), line.c_str(), 0);
    }

    // Header is complete. Now we need to generate the SQ mapping data.
    m_sq_mapping.clear();
    for (size_t hdr_idx = 0; hdr_idx < m_ref_info_lut.size(); ++hdr_idx) {
        const auto& info_lut = m_ref_info_lut[hdr_idx];
        std::vector<uint32_t> hdr_mapping(info_lut.size(), std::numeric_limits<uint32_t>::max());
        for (const auto& entry : info_lut) {
            const auto& key = entry.first;
            const auto& info = entry.second;
            auto original_sq_idx = info.index;
            if (original_sq_idx >= hdr_mapping.size()) {
                // This should be impossible, and would indicate a bug in this code.
                throw std::range_error("SQ index out of bounds when merging headers.");
            }
            auto new_sq_idx = new_sq_order.at(key);
            hdr_mapping[original_sq_idx] = new_sq_idx;
        }
        // Sanity check.
        for (auto n : hdr_mapping) {
            if (n == std::numeric_limits<uint32_t>::max()) {
                throw std::logic_error("Inconsistent SQ data in header merging.");
            }
        }
        m_sq_mapping.emplace_back(std::move(hdr_mapping));
    }
}

bool MergeHeaders::add_rg(const std::string& read_group_id, std::string read_group_line) {
    // Add the RG_line to the LUT or error if it a different record already exists
    auto entry = m_read_group_lut.find(read_group_id);
    if (entry == m_read_group_lut.end()) {
        m_read_group_lut[read_group_id] = std::move(read_group_line);
    } else {
        if (entry->second != read_group_line) {
            return false;
        }
    }
    return true;
};

bool MergeHeaders::add_rg(const std::string& read_group_id,
                          const ReadGroup& read_group,
                          const std::map<std::string, std::string>& additional_tags) {
    const auto additional_tag_str = kv_to_tag_string(additional_tags);
    return add_rg(read_group_id, utils::format_read_group_header_line(read_group, read_group_id,
                                                                      additional_tag_str));
}

}  // namespace dorado::utils
