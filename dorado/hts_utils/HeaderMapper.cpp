#include "hts_utils/HeaderMapper.h"

#include "hts_utils/FastxSequentialReader.h"
#include "hts_utils/MergeHeaders.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/fastq_tags.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_types.h"
#include "hts_utils/sequence_file_format.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

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

void assign_not_empty(std::string& attr_target, const std::string_view maybe_value) {
    if (!maybe_value.empty()) {
        attr_target = maybe_value;
    }
};

}  // anonymous namespace

namespace dorado::utils {

HeaderMapper::HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment)
        : m_strip_alignment(strip_alignment),
          m_read_group_to_attributes(std::make_shared<HeaderMapper::AttributeMap>()),
          m_merged_headers_map(std::make_shared<HeaderMapper::HeaderMap>()) {
    m_merged_headers_map->emplace(m_fallback_read_attrs,
                                  std::make_unique<MergeHeaders>(m_strip_alignment));
    process(inputs);
}

void HeaderMapper::process(const std::vector<std::filesystem::path>& inputs) {
    for (const auto& input : inputs) {
        if (hts_io::parse_sequence_format(input) == hts_io::SequenceFormatType::FASTQ ||
            hts_io::parse_sequence_format(input) == hts_io::SequenceFormatType::FASTA) {
            if (!m_fastq_runtime_warning_issued) {
                m_fastq_runtime_warning_issued = true;
                spdlog::warn("Mapping headers from FASTQ files. This might take some time.");
            }
            process_fastx(input);
        } else {
            process_bam(input);
        }
    }

    // Finalize the headers
    for (const auto& [_, merged_header_ptr] : *m_merged_headers_map) {
        merged_header_ptr->finalize_merge();
    }
};

void HeaderMapper::process_fastx(const std::filesystem::path& path) {
    spdlog::trace("HeaderMapper::process_fastx processing '{}'", path.string());

    hts_io::FastxSequentialReader reader(path);
    hts_io::FastxRecord record;

    std::unordered_map<std::string, HtsData::ReadAttributes> rg_id_to_attrs_lut;
    const auto& fallback_merged_header = m_merged_headers_map->at(m_fallback_read_attrs);

    bool debug_msg_issued = false;
    while (reader.get_next(record)) {
        // Check if the tags are HTS-style and parse them.
        ReadGroupData rg_data = dorado::utils::parse_rg_from_hts_tags(record.comment);

        if (!rg_data.found) {
            if (!debug_msg_issued) {
                debug_msg_issued = true;
                spdlog::debug("FASTQ record missing read group data in file '{}'", path.string());
            }
            fallback_merged_header->add_header(sam_hdr_init(), path.string(),
                                               m_fallback_read_attrs.protocol_run_id);
            continue;
        }

        auto [it, inserted] = m_read_group_to_attributes->try_emplace(rg_data.id);
        if (!inserted) {
            continue;
        }

        HtsData::ReadAttributes& attrs = it->second;
        assign_not_empty(attrs.flowcell_id, rg_data.data.flowcell_id);
        assign_not_empty(attrs.position_id, rg_data.data.device_id);
        assign_not_empty(attrs.sample_id, rg_data.data.sample_id);
        assign_not_empty(attrs.protocol_run_id, rg_data.data.run_id);
        assign_not_empty(attrs.experiment_id, rg_data.data.experiment_id);

        if (!rg_data.data.exp_start_time.empty()) {
            attrs.protocol_start_time_ms =
                    utils::get_unix_time_ms_from_string_timestamp(rg_data.data.exp_start_time);
        }

        // Create a new empty merged header
        auto& merged_header_ptr = (*m_merged_headers_map)[attrs];
        if (!merged_header_ptr) {
            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
            merged_header_ptr->add_header(sam_hdr_init(), path.string(), rg_data.id);
        }
        merged_header_ptr->add_rg(rg_data.id, rg_data.data);
    }
}

void HeaderMapper::process_bam(const std::filesystem::path& path) {
    spdlog::trace("HeaderMapper::process_bam processing '{}'", path.string());

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
    const auto rg_to_attrs_lut = get_read_attrs_by_id(header_lines);

    auto& merged_headers = *m_merged_headers_map;

    // Add the new read attrs and merge the headers for each output
    // file only including the read groups that will be used.
    for (const auto& [read_group_id, read_attrs] : rg_to_attrs_lut) {
        (*m_read_group_to_attributes)[read_group_id] = read_attrs;

        auto& merged_header_ptr = merged_headers[read_attrs];
        if (!merged_header_ptr) {
            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
        }
        merged_header_ptr->add_header(header.get(), path.string(), read_group_id);
    }
}

std::unordered_map<std::string, HtsData::ReadAttributes> HeaderMapper::get_read_attrs_by_id(
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
        auto [it, inserted] = rg_id_to_attrs_lut.try_emplace(rg_id);
        if (!inserted) {
            continue;
        }

        HtsData::ReadAttributes& attrs = it->second;
        attrs.protocol_start_time_ms = parse_DT_tag(tags);

        assign_not_empty(attrs.flowcell_id, get_tag("PU", tags));
        assign_not_empty(attrs.sample_id, get_tag("LB", tags));
        // TODO: position_id is not in the specification yet

        const auto& tag_it = tags.find("DS");
        const std::string ds = (tag_it != std::end(tags)) ? tag_it->second : "";
        const auto ds_tokens = tokenize(ds, ' ');
        assign_not_empty(attrs.protocol_run_id, parse_DS_tag_key(ds_tokens, "runid="));
        // TODO: experiment_id and acquisition_id are not in the specification yet
    }

    return rg_id_to_attrs_lut;
};

void HeaderMapper::modify_headers(const Modifier& modifier) const {
    for (const auto& [_, merged_header_ptr] : *m_merged_headers_map) {
        modifier(merged_header_ptr->get_merged_header());
    }
};

const HtsData::ReadAttributes& HeaderMapper::get_read_attributes(const bam1_t* record) const {
    // Get the read group ID from the record
    const std::string read_group = utils::get_read_group_tag(record);
    if (read_group.empty()) {
        return m_fallback_read_attrs;
    }

    // Lookup the ReadAttributes for this read_group
    const auto attr_it = m_read_group_to_attributes->find(read_group);
    if (attr_it == m_read_group_to_attributes->cend()) {
        return m_fallback_read_attrs;
    }

    return attr_it->second;
};

const MergeHeaders& HeaderMapper::get_merged_header(const HtsData::ReadAttributes& attrs) const {
    // Lookup the merged header for these ReadAttributes
    const auto header_it = m_merged_headers_map->find(attrs);
    if (header_it == m_merged_headers_map->cend()) {
        spdlog::error("Read group attributes were not found in mapped headers: runid='{}'.",
                      attrs.protocol_run_id);
        throw std::runtime_error("Merged header not found");
    }

    return *header_it->second;
};

SamHdrPtr HeaderMapper::get_shared_merged_header(bool strip_alignments) const {
    MergeHeaders merged(strip_alignments);
    for (const auto& [read_attrs, header] : *get_merged_headers_map()) {
        merged.add_header(header->get_merged_header(), "-");
    }
    merged.finalize_merge();
    return SamHdrPtr(sam_hdr_dup(merged.get_merged_header()));
};

}  // namespace dorado::utils
