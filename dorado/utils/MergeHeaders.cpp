#include "MergeHeaders.h"

#include "utils/bam_utils.h"

#include <htslib/sam.h>

#include <limits>
#include <numeric>
#include <sstream>

namespace {
void update_and_add_pg_line(sam_hdr_t* hdr, const std::string& key, std::string line) {
    std::string new_id = sam_hdr_pg_id(hdr, key.c_str());
    auto pos = line.find(key);
    line.replace(pos, key.size(), new_id);
    sam_hdr_add_lines(hdr, line.c_str(), 0);
}

}  // anonymous namespace

namespace dorado::utils {

MergeHeaders::MergeHeaders(bool strip_alignment) : m_strip_alignment(strip_alignment) {}

std::string MergeHeaders::add_header(sam_hdr_t* hdr, const std::string& filename) {
    if (!m_strip_alignment) {
        auto res = check_and_add_ref_data(hdr);
        if (res == -1) {
            return "Error merging header " + filename + ". Invalid SQ line in header.";
        }
        if (res == -2) {
            return "Error merging header " + filename + ". SQ lines are incompatible.";
        }
    }

    auto res = check_and_add_rg_data(hdr);
    if (res == -1) {
        return "Error merging header " + filename + ". Invalid RG line in header.";
    }
    if (res == -2) {
        return "Error merging header " + filename + ". RG lines are incompatible.";
    }

    res = add_pg_data(hdr);
    if (res < 0) {
        return "Error merging header " + filename + ". Invalid PG line in header.";
    }

    add_other_lines(hdr);
    return std::string();
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

int MergeHeaders::check_and_add_ref_data(sam_hdr_t* hdr) {
    std::map<std::string, RefInfo> ref_map;
    int nrefs = sam_hdr_nref(hdr);
    for (int i = 0; i < nrefs; ++i) {
        KString line_wrapper(1000000);
        auto line_data = line_wrapper.get();
        auto res = sam_hdr_find_line_pos(hdr, "SQ", i, &line_data);
        if (res < 0) {
            return -1;
        }
        auto ref_name = sam_hdr_line_name(hdr, "SQ", i);
        if (!ref_name) {
            return -1;
        }
        std::string key(ref_name);
        std::string line(ks_str(&line_data));
        RefInfo info{uint32_t(i), line};
        ref_map[key] = std::move(info);
        auto entry = m_ref_lut.find(key);
        if (entry == m_ref_lut.end()) {
            m_ref_lut[key] = line;
        } else {
            if (line != entry->second) {
                return -2;
            }
        }
    }
    m_ref_info_lut.emplace_back(std::move(ref_map));
    return 0;
}

int MergeHeaders::check_and_add_rg_data(sam_hdr_t* hdr) {
    int num_lines = sam_hdr_count_lines(hdr, "RG");
    for (int i = 0; i < num_lines; ++i) {
        auto idp = sam_hdr_line_name(hdr, "RG", i);
        if (!idp) {
            return -1;
        }
        KString line_wrapper(1000000);
        auto line_data = line_wrapper.get();
        auto res = sam_hdr_find_line_pos(hdr, "RG", i, &line_data);
        if (res < 0) {
            return -1;
        }
        std::string key(idp);
        std::string line(ks_str(&line_data));
        auto entry = m_read_group_lut.find(key);
        if (entry == m_read_group_lut.end()) {
            m_read_group_lut[key] = std::move(line);
        } else {
            if (entry->second != line) {
                return -2;
            }
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
        const auto& line = entry.second;
        sam_hdr_add_lines(m_merged_header.get(), line.c_str(), 0);
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

}  // namespace dorado::utils
