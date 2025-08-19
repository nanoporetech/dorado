#include "hts_utils/HeaderMapper.h"

#include "hts_utils/MergeHeaders.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/fastq_reader.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_types.h"
#include "spdlog/spdlog.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using namespace dorado;

std::string get_tag(const std::string& tag,
                    const std::unordered_map<std::string, std::string>& tags) {
    const auto& it = tags.find(tag);
    const std::string value = (it != std::end(tags)) ? it->second : "";
    return value;
}

std::vector<std::string> tokenize(const std::string& line, const char delimiter) {
    if (line.empty()) {
        return {};
    }
    std::istringstream stream{line};
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

int64_t parse_DT_tag(const std::unordered_map<std::string, std::string>& tags) {
    // Expected ISO8601 timestamp
    const auto dt = get_tag("DT", tags);
    if (dt.empty()) {
        return 0;
    }
    const int64_t datetime_ms = utils::get_unix_time_ms_from_string_timestamp(dt);
    return datetime_ms;
}

std::string_view parse_DS_tag_key(const std::vector<std::string>& tokens, const std::string& key) {
    if (tokens.size() < 2 || key.empty()) {
        return "";
    }

    for (const auto& token : tokens) {
        auto token_view = std::string_view(token);
        if (token_view.substr(0, key.size()) == key) {
            return token_view.substr(key.size());
        }
    }

    return "";
}

}  // anonymous namespace

namespace dorado::utils {

HeaderMapper::HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment)
        : m_strip_alignment(strip_alignment),
          m_read_group_to_attributes(std::make_shared<HeaderMapper::AttributeMap>()),
          m_merged_headers(std::make_shared<HeaderMapper::HeaderMap>()) {
    process(inputs);
}

void HeaderMapper::process(const std::vector<std::filesystem::path>& inputs) {
    for (const auto& input : inputs) {
        if (is_fastq(input.string())) {
            process_fastq(input);
        } else {
            process_bam(input);
        }
    }

    // Finalize the headers
    for (const auto& [_, merged_header_ptr] : *m_merged_headers) {
        merged_header_ptr->finalize_merge();
    }
};

void HeaderMapper::process_bam(const std::filesystem::path& path) {
    auto file = dorado::HtsFilePtr(hts_open(path.string().c_str(), "r"));
    if (!file) {
        spdlog::error("Failed to open file: '{}'.", path.string());
        throw std::runtime_error("Could not open file for mapping");
    }
    dorado::SamHdrPtr header(sam_hdr_read(file.get()));
    if (!header) {
        spdlog::error("Failed to read header from file: '{}'.", path.string());
        throw std::runtime_error("Could not open header for mapping");
    }

    const auto header_lines = utils::parse_header(*header.get(), {utils::HeaderLineType::RG});

    // Map read group ids to ReadAttributes (struct containing file naming parameters)
    const auto rg_to_attrs_lut = get_read_attrs_lut(header_lines);

    auto& read_group_to_attributes = *m_read_group_to_attributes;
    auto& merged_headers = *m_merged_headers;

    // Add the new read attrs and merge the headers for each output
    // file only including the read groups that will be used.
    for (const auto& [read_group_id, read_attrs] : rg_to_attrs_lut) {
        read_group_to_attributes[read_group_id] = read_attrs;

        auto& merged_header_ptr = merged_headers[read_attrs];
        if (!merged_header_ptr) {
            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
        }
        merged_header_ptr->add_header(header.get(), path.string(), read_group_id);
    }
}

std::unordered_map<std::string, HtsData::ReadAttributes> HeaderMapper::get_read_attrs_lut(
        const std::vector<utils::HeaderLineData>& rg_lines) {
    // Example RG line:
    // @RG	ID:e705d8cfbbe8a6bc43a865c71ace09553e8f15cd_dna_r10.4.1_e8.2_400bps_hac@v5.0.0
    //  DT:2022-10-18T10:38:07.247961+00:00
    //  DS:runid=e705d8cfbbe8a6bc43a865c71ace09553e8f15cd ...
    //  LB:PCR_zymo PL:ONT   PM:4A  PU:PAM93185
    std::unordered_map<std::string, HtsData::ReadAttributes> rg_id_to_attrs_lut;
    for (const auto& rg_line : rg_lines) {
        if (rg_line.header_type != utils::HeaderLineType::RG) {
            continue;
        }

        // Convert all tags into a lookup.
        const std::unordered_map<std::string, std::string> tags = [&]() {
            std::unordered_map<std::string, std::string> tag_map;
            for (const auto& [key, value] : rg_line.tags) {
                tag_map[key] = value;
            }
            return tag_map;
        }();

        // Parse the read group ID.
        const std::string rg_id = get_tag("ID", tags);
        if (rg_id.empty()) {
            continue;
        }

        // Get the read attributes for this read group or create a new default one.
        HtsData::ReadAttributes& attrs = rg_id_to_attrs_lut[rg_id];
        attrs.protocol_start_time_ms = parse_DT_tag(tags);

        const auto flowcell_id = get_tag("PU", tags);
        if (!flowcell_id.empty()) {
            attrs.flowcell_id = flowcell_id;
        }

        const auto position_id = get_tag("PM", tags);
        if (!position_id.empty()) {
            attrs.position_id = position_id;
        }

        const auto sample_id = get_tag("LB", tags);
        if (!sample_id.empty()) {
            attrs.sample_id = sample_id;
        }

        {
            const auto& it = tags.find("DS");
            const std::string ds = (it != std::end(tags)) ? it->second : "";

            const auto ds_tokens = tokenize(ds, ' ');
            attrs.protocol_run_id = parse_DS_tag_key(ds_tokens, "runid=");
            // TODO: These fields are not in the specification yet
            // attrs.experiment_id = parse_DS_tag_key(ds_tokens, "experiment_name");
            // attrs.acquisition_id = parse_DS_tag_key(ds_tokens, "acquisition_id");
        }
    }

    return rg_id_to_attrs_lut;
};

void HeaderMapper::process_fastq([[maybe_unused]] const std::filesystem::path& path) {
    throw std::logic_error("HeaderMapper::process_fastq is not implemented");
};

void HeaderMapper::modify_headers(const Modifier& modifier) const {
    for (const auto& [_, merged_header_ptr] : *m_merged_headers) {
        modifier(merged_header_ptr->get_merged_header());
    }
};

const HtsData::ReadAttributes& HeaderMapper::get_read_attributes(const bam1_t* record) const {
    // Get the read group ID from the record
    const std::string read_group = utils::get_read_group_tag(record);
    if (read_group.empty()) {
        spdlog::error("Read group of htslib record is unexpectedly empty");
        throw std::runtime_error("Invalid record edit - Read group is Empty");
    }

    // Lookup the ReadAttributes for this read_group
    const auto attr_it = m_read_group_to_attributes->find(read_group);
    if (attr_it == m_read_group_to_attributes->cend()) {
        spdlog::error("Read group was not found in mapped attributes: '{}'", read_group);
        throw std::runtime_error("Invalid record edit - Attributes not found");
    }

    return attr_it->second;
};
const MergeHeaders& HeaderMapper::get_merged_header(const bam1_t* record) const {
    return get_merged_header(get_read_attributes(record));
}

const MergeHeaders& HeaderMapper::get_merged_header(const HtsData::ReadAttributes& attrs) const {
    // Lookup the merged header for these ReadAttributes
    const auto header_it = m_merged_headers->find(attrs);
    if (header_it == m_merged_headers->cend()) {
        spdlog::error("Read group attributes were not found in mapped headers: runid='{}'.",
                      attrs.protocol_run_id);
        throw std::runtime_error("Merged header not found");
    }

    return *header_it->second;
};

}  // namespace dorado::utils
