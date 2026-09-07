#include "hts_utils/HeaderMapper.h"

#include "hts_utils/FastxSequentialReader.h"
#include "hts_utils/MergeHeaders.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/fastq_tags.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_types.h"
#include "utils/SampleSheet.h"
#include "utils/barcode_kits.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using namespace dorado;

std::string get_barcode_sequence(const std::string& barcode_name) {
    const auto& barcode_sequences = barcode_kits::get_barcodes();
    const auto sequence_itr = barcode_sequences.find(barcode_name);
    if (sequence_itr != barcode_sequences.cend()) {
        return sequence_itr->second;
    }
    throw std::runtime_error("Unrecognised barcode name: " + barcode_name);
}

std::string get_tag(const std::string& tag,
                    const std::unordered_map<std::string, std::string>& tags) {
    const auto& it = tags.find(tag);
    return (it != std::end(tags)) ? it->second : "";
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
    const auto datetime_ms = utils::get_unix_time_ms_from_string_timestamp(dt);
    return datetime_ms.value_or(0);
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

std::string resolve_output_read_group_id(const utils::MergeHeaders& merged_header,
                                         const std::string& filename,
                                         const std::string& read_group_id) {
    if (const auto remapped_read_group_id =
                merged_header.get_remapped_read_group_id(filename, read_group_id);
        remapped_read_group_id) {
        return *remapped_read_group_id;
    }
    return read_group_id;
}

}  // anonymous namespace

namespace dorado::utils {

HeaderMapper::HeaderMapper(const std::unordered_map<std::string, ReadGroup>& read_groups,
                           std::optional<std::string> kit_name,
                           const utils::SampleSheet* const sample_sheet)
        : HeaderMapper(std::move(kit_name), sample_sheet, true) {
    process(read_groups);
    finalize_merge();
}

HeaderMapper::HeaderMapper(const std::vector<std::filesystem::path>& inputs,
                           std::optional<std::string> kit_name,
                           const utils::SampleSheet* const sample_sheet,
                           bool strip_alignment)
        : HeaderMapper(std::move(kit_name), sample_sheet, strip_alignment) {
    process(inputs);
    finalize_merge();
}

HeaderMapper::HeaderMapper(std::optional<std::string> kit_name,
                           const utils::SampleSheet* const sample_sheet,
                           bool strip_alignment)
        : m_kit_name(std::move(kit_name)),
          m_sample_sheet(sample_sheet),
          m_strip_alignment(strip_alignment),
          m_merged_headers_map(std::make_shared<HeaderMapper::HeaderMap>()) {
    if (m_kit_name) {
        m_fallback_read_attrs.barcode_id = UNCLASSIFIED_STR;
        m_read_group_to_attributes[""] = m_fallback_read_attrs;
    }
    m_merged_headers_map->emplace(m_fallback_read_attrs,
                                  std::make_unique<MergeHeaders>(m_strip_alignment));
}

void HeaderMapper::process(const std::unordered_map<std::string, ReadGroup>& read_groups) {
    m_has_barcodes = m_kit_name.has_value();
    auto& merged_headers = *m_merged_headers_map;
    for (auto [id, read_group] : read_groups) {
        HtsData::ReadAttributes attrs;
        assign_not_empty(attrs.flowcell_id, read_group.flowcell_id);
        assign_not_empty(attrs.position_id, read_group.position_id);
        assign_not_empty(attrs.sample_id, read_group.sample_id);
        assign_not_empty(attrs.protocol_run_id, read_group.run_id);
        assign_not_empty(attrs.experiment_id, read_group.experiment_id);
        attrs.trim_flags = read_group.trim_flags;

        if (m_kit_name) {
            read_group.barcode_id = UNCLASSIFIED_STR;
            attrs.barcode_id = UNCLASSIFIED_STR;
            read_group.barcode_alias = "";
            attrs.barcode_alias = "";
        }

        if (!read_group.exp_start_time.empty()) {
            const auto protocol_start_time_ms =
                    utils::get_unix_time_ms_from_string_timestamp(read_group.exp_start_time);
            if (protocol_start_time_ms) {
                attrs.protocol_start_time_ms = protocol_start_time_ms.value();
            }
        }

        auto& merged_header_ptr = merged_headers[attrs];
        if (!merged_header_ptr) {
            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
        }

        merged_header_ptr->add_rg(id, read_group, {});
        m_read_group_to_attributes[id] = attrs;
    }
}

void HeaderMapper::process(const std::vector<std::filesystem::path>& inputs) {
    for (const auto& input : inputs) {
        if (hts_io::parse_sequence_format(input) == hts_io::SequenceFormatType::FASTQ ||
            hts_io::parse_sequence_format(input) == hts_io::SequenceFormatType::FASTA) {
            if (!std::exchange(m_fastq_runtime_warning_issued, true)) {
                spdlog::warn(
                        "Mapping headers from FASTQ files. This might take some time. Using BAM "
                        "files as input is recommended as this avoids FASTQ header mapping.");
            }
            process_fastx(input);
        } else {
            process_bam(input);
        }
    }
}

void HeaderMapper::process_fastx(const std::filesystem::path& path) {
    spdlog::trace("HeaderMapper::process_fastx processing '{}'", path.string());
    const auto path_string = path.string();

    hts_io::FastxSequentialReader reader(path);
    hts_io::FastxRecord record;

    const auto& fallback_merged_header = m_merged_headers_map->at(m_fallback_read_attrs);

    bool debug_msg_issued = false;

    auto& merged_headers = *m_merged_headers_map;
    std::unordered_map<std::string, HtsData::ReadAttributes> rg_to_attrs_lut;

    SamHdrPtr hdr(sam_hdr_init());
    while (reader.get_next(record)) {
        // Check if the tags are HTS-style and parse them.
        ReadGroupData rg_data = dorado::utils::parse_rg_from_hts_tags(record.comment);

        if (!rg_data.found) {
            if (!debug_msg_issued) {
                debug_msg_issued = true;
                spdlog::debug("FASTQ record missing read group data in file '{}'", path.string());
            }
            fallback_merged_header->add_header(hdr.get(), path_string,
                                               m_fallback_read_attrs.protocol_run_id);
            continue;
        }

        if (m_kit_name) {
            // convert to unclassified - we'll use this to create all the barcoded headers later
            std::string_view alias = rg_data.data.barcode_alias.empty()
                                             ? rg_data.data.barcode_id
                                             : rg_data.data.barcode_alias;
            if (!alias.empty() && alias != UNCLASSIFIED_STR) {
                if (auto index = rg_data.id.find(alias); index != rg_data.id.npos && index != 0) {
                    rg_data.id = rg_data.id.substr(0, index - 1);
                }
            }
            rg_data.data.barcode_id = UNCLASSIFIED_STR;
            rg_data.data.barcode_alias.clear();
        }

        auto [it, inserted] = rg_to_attrs_lut.try_emplace(rg_data.id);
        if (!inserted) {
            continue;
        }

        m_has_barcodes |= !rg_data.data.barcode_id.empty();
        m_has_barcodes |= !rg_data.data.barcode_alias.empty();

        HtsData::ReadAttributes& attrs = it->second;
        assign_not_empty(attrs.flowcell_id, rg_data.data.flowcell_id);
        // TODO: position_id is not in the specification yet
        // assign_not_empty(attrs.position_id, rg_data.data.position_id);
        assign_not_empty(attrs.sample_id, rg_data.data.sample_id);
        assign_not_empty(attrs.protocol_run_id, rg_data.data.run_id);
        assign_not_empty(attrs.experiment_id, rg_data.data.experiment_id);
        assign_not_empty(attrs.barcode_id, rg_data.data.barcode_id);
        assign_not_empty(attrs.barcode_alias, rg_data.data.barcode_alias);
        attrs.trim_flags = rg_data.data.trim_flags;

        if (attrs.barcode_alias == attrs.barcode_id || attrs.barcode_alias == UNCLASSIFIED_STR) {
            // File headers may have both the barcode and alias set to the same value
            // But when we classify we leave the alias blank if it is unused, so clear it here to match
            attrs.barcode_alias.clear();
        }

        if (!rg_data.data.exp_start_time.empty()) {
            const auto protocol_start_time_ms =
                    utils::get_unix_time_ms_from_string_timestamp(rg_data.data.exp_start_time);
            if (protocol_start_time_ms) {
                attrs.protocol_start_time_ms = protocol_start_time_ms.value();
            }
        }

        auto& merged_header_ptr = merged_headers[attrs];
        if (!merged_header_ptr) {
            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
        }

        std::map<std::string, std::string> kv_pairs;
        if (!(attrs.barcode_id.empty() || attrs.barcode_id == UNCLASSIFIED_STR)) {
            kv_pairs = {
                    {"SM", attrs.barcode_id},
                    {"al", attrs.barcode_alias.empty() ? attrs.barcode_id : attrs.barcode_alias},
                    // BC and bk tags are not present in fastq files
            };
        }
        const auto output_read_group_id = merged_header_ptr->add_rg_with_remap(
                path_string, rg_data.id, rg_data.data, kv_pairs);
        m_output_read_group_ids_by_file[path_string][rg_data.id] = output_read_group_id;
        if (output_read_group_id != rg_data.id) {
            rg_data.id = output_read_group_id;
        }
    }

    // Add the new read attrs and merge the headers for each output
    // file only including the read groups that will be used.
    for (const auto& [read_group_id, read_attrs] : rg_to_attrs_lut) {
        const auto output_read_group_id = get_output_read_group_id(path_string, read_group_id);
        m_read_group_to_attributes[output_read_group_id] = read_attrs;
        auto& merge_header = merged_headers[read_attrs];
        merge_header->add_header(hdr.get(), path_string, output_read_group_id);
    }
}

void HeaderMapper::process_bam(const std::filesystem::path& path) {
    spdlog::trace("HeaderMapper::process_bam processing '{}'", path.string());
    const auto path_string = path.string();

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

    if (rg_to_attrs_lut.empty()) {
        // No RG lines in the BAM header: route this file through the fallback merged header.
        merged_headers.at(m_fallback_read_attrs)->add_header(header.get(), path_string, "");
        return;
    }

    // Add the new read attrs and merge the headers for each output
    // file only including the read groups that will be used.
    for (auto [id, read_attrs] : rg_to_attrs_lut) {
        std::string read_group_id = id;
        if (m_kit_name) {
            // convert to unclassified - we'll use this to create all the barcoded headers later
            std::string_view alias = read_attrs.barcode_alias.empty() ? read_attrs.barcode_id
                                                                      : read_attrs.barcode_alias;
            if (!alias.empty() && alias != UNCLASSIFIED_STR) {
                if (auto index = read_group_id.find(alias);
                    index != read_group_id.npos && index != 0) {
                    read_group_id = read_group_id.substr(0, index - 1);
                }
            }
            read_attrs.barcode_id = UNCLASSIFIED_STR;
            read_attrs.barcode_alias.clear();
        }
        m_has_barcodes |= !read_attrs.barcode_id.empty();
        m_has_barcodes |= !read_attrs.barcode_alias.empty();
        {
            auto& merged_header_ptr = merged_headers[read_attrs];
            if (!merged_header_ptr) {
                merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
            }
            merged_header_ptr->add_header(header.get(), path_string, read_group_id);
            const auto output_read_group_id =
                    resolve_output_read_group_id(*merged_header_ptr, path_string, read_group_id);
            m_output_read_group_ids_by_file[path_string][read_group_id] = output_read_group_id;
            m_read_group_to_attributes[output_read_group_id] = read_attrs;
        }
    }
}

void HeaderMapper::add_barcodes() {
    m_has_barcodes = true;
    auto& merged_headers = *m_merged_headers_map;

    // ensure we have an unclassified base entry for each barcoded entry
    AttributeMap unclassified_entries;
    for (auto [id, attrs] : m_read_group_to_attributes) {
        const auto& base_header = merged_headers[attrs];
        std::string rg_id = id;
        if (attrs.barcode_id != UNCLASSIFIED_STR) {
            std::string_view alias =
                    attrs.barcode_alias.empty() ? attrs.barcode_id : attrs.barcode_alias;
            if (auto index = rg_id.find(alias); index != rg_id.npos && index != 0) {
                rg_id = rg_id.substr(0, index - 1);
            }
            attrs.barcode_id = UNCLASSIFIED_STR;
            attrs.barcode_alias = "";
        }

        auto& merge_header = merged_headers[attrs];
        if (!merge_header) {
            SamHdrPtr header(sam_hdr_dup(base_header->get_merged_header()));
            sam_hdr_update_line(header.get(), "RG", "ID", id.c_str(), "ID", rg_id.c_str(), nullptr);
            sam_hdr_remove_tag_id(header.get(), "RG", "ID", rg_id.c_str(), "SM");
            sam_hdr_remove_tag_id(header.get(), "RG", "ID", rg_id.c_str(), "al");
            sam_hdr_remove_tag_id(header.get(), "RG", "ID", rg_id.c_str(), "bk");
            sam_hdr_remove_tag_id(header.get(), "RG", "ID", rg_id.c_str(), "BC");
            merge_header = std::make_unique<MergeHeaders>(m_strip_alignment);
            merge_header->add_header(header.get(), "", rg_id);
            merge_header->finalize_merge();
        }

        unclassified_entries[rg_id] = std::move(attrs);
    }

    std::unordered_map<std::string, HtsData::ReadAttributes> rg_to_attrs_lut;
    for (auto [read_group_id, read_attrs] : unclassified_entries) {
        const auto& base_header = merged_headers[read_attrs];
        const auto& kit_info_map = barcode_kits::get_kit_infos();
        const auto& kit_info = kit_info_map.at(*m_kit_name);

        auto add_bc_header = [&](const std::string& outer_barcode_name,
                                 const std::string& inner_barcode_name,
                                 const std::string& normalized_barcode_name,
                                 const std::string& standard_barcode_name) {
            std::string alias;
            if (m_sample_sheet) {
                if (!m_sample_sheet->barcode_is_permitted(normalized_barcode_name)) {
                    return;
                }
                alias = m_sample_sheet->get_alias(read_attrs.flowcell_id, read_attrs.position_id,
                                                  read_attrs.experiment_id,
                                                  normalized_barcode_name);
                read_attrs.barcode_alias = alias;
            }

            read_attrs.barcode_alias = alias;
            read_attrs.barcode_id = normalized_barcode_name;

            auto& merged_header_ptr = merged_headers[read_attrs];
            if (merged_header_ptr) {
                // we already found this barcode header in an input file
                return;
            }

            SamHdrPtr header(sam_hdr_dup(base_header->get_merged_header()));
            auto new_read_group_id =
                    read_group_id + "_" + (alias.empty() ? standard_barcode_name : alias);
            if (!read_group_id.empty()) {
                std::string barcode_sequence = get_barcode_sequence(outer_barcode_name);
                if (!inner_barcode_name.empty()) {
                    barcode_sequence += "-" + get_barcode_sequence(inner_barcode_name);
                }
                sam_hdr_update_line(header.get(), "RG", "ID", read_group_id.c_str(), "SM",
                                    normalized_barcode_name.c_str(), "al",
                                    alias.empty() ? normalized_barcode_name.c_str() : alias.c_str(),
                                    "ID", new_read_group_id.c_str(), "bk", m_kit_name->c_str(),
                                    "BC", barcode_sequence.c_str(), nullptr);
            }
            rg_to_attrs_lut[new_read_group_id] = read_attrs;

            merged_header_ptr = std::make_unique<MergeHeaders>(m_strip_alignment);
            merged_header_ptr->add_header(header.get(), "", new_read_group_id);
            merged_header_ptr->finalize_merge();
        };

        for (const auto& barcode_name : kit_info.barcodes) {
            const auto normalized_barcode_name = barcode_kits::normalize_barcode_name(barcode_name);
            const auto standard_barcode_name =
                    barcode_kits::generate_standard_barcode_name(*m_kit_name, barcode_name);
            if (kit_info.barcodes_inner1.empty()) {
                add_bc_header(barcode_name, "", normalized_barcode_name, standard_barcode_name);
            } else {
                for (const auto& inner_barcode_name : kit_info.barcodes_inner1) {
                    const auto normalized_inner_barcode_name =
                            barcode_kits::normalize_barcode_name(inner_barcode_name);
                    // Note that we don't include the barcode kit again in the standard barcode name for dual.
                    add_bc_header(barcode_name, inner_barcode_name,
                                  std::string(normalized_barcode_name)
                                          .append("_")
                                          .append(normalized_inner_barcode_name),
                                  std::string(standard_barcode_name)
                                          .append("_")
                                          .append(normalized_inner_barcode_name));
                }
            }
        }
    }
    m_read_group_to_attributes.merge(unclassified_entries);
    m_read_group_to_attributes.merge(rg_to_attrs_lut);
}

void HeaderMapper::finalize_merge() {
    // Finalize the headers
    for (const auto& [_, merged_header_ptr] : *m_merged_headers_map) {
        merged_header_ptr->finalize_merge();
    }

    if (m_kit_name) {
        add_barcodes();
    }
}

std::unordered_map<std::string, HtsData::ReadAttributes> HeaderMapper::get_read_attrs_by_id(
        const std::vector<utils::HeaderLineData>& rg_lines) {
    // Example RG line:
    // @RG	ID:e705d8cfbbe8a6bc43a865c71ace09553e8f15cd_dna_r10.4.1_e8.2_400bps_hac@v5.0.0
    //  DT:2022-10-18T10:38:07.247961+00:00
    //  DS:runid=e705d8cfbbe8a6bc43a865c71ace09553e8f15cd ...
    //  LB:PCR_zymo PL:ONT   PM:MN12345  PU:PAM93185
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
        assign_not_empty(attrs.barcode_id, get_tag("SM", tags));
        assign_not_empty(attrs.barcode_alias, get_tag("al", tags));
        if (attrs.barcode_alias == attrs.barcode_id || attrs.barcode_alias == UNCLASSIFIED_STR) {
            // File headers may have both the barcode and alias set to the same value
            // But when we classify we leave the alias blank if it is unused, so clear it here to match
            attrs.barcode_alias.clear();
        }
        attrs.trim_flags = TrimFlags::from_string(get_tag("tm", tags));

        // TODO: position_id is not in the specification yet

        const auto& tag_it = tags.find("DS");
        const std::string ds = (tag_it != std::end(tags)) ? tag_it->second : "";
        const auto ds_tokens = tokenize(ds, ' ');
        assign_not_empty(attrs.protocol_run_id, parse_DS_tag_key(ds_tokens, "runid="));
        assign_not_empty(attrs.experiment_id, parse_DS_tag_key(ds_tokens, "experiment_id="));
        // TODO: acquisition_id is not in the specification yet
    }

    return rg_id_to_attrs_lut;
};

void HeaderMapper::modify_headers(const Modifier& modifier) const {
    for (const auto& [_, merged_header_ptr] : *m_merged_headers_map) {
        modifier(merged_header_ptr->get_merged_header());
    }
};

std::string HeaderMapper::get_output_read_group_id(const std::string& filename,
                                                   const std::string& read_group_id) const {
    const auto file_it = m_output_read_group_ids_by_file.find(filename);
    if (file_it == m_output_read_group_ids_by_file.cend()) {
        return read_group_id;
    }

    const auto rg_it = file_it->second.find(read_group_id);
    if (rg_it == file_it->second.cend()) {
        return read_group_id;
    }

    return rg_it->second;
}

const HtsData::ReadAttributes& HeaderMapper::get_read_attributes(const bam1_t* record) const {
    // Get the read group ID from the record
    std::string read_group = utils::get_read_group_tag(record);
    if (read_group.empty()) {
        return m_fallback_read_attrs;
    }

    if (m_kit_name) {
        // lookup by the unclassified read group if we're going to be barcoding
        std::string alias;
        if (const auto al_tag = bam_aux_get(record, "al"); al_tag != nullptr) {
            alias = bam_aux2Z(al_tag);
        } else if (const auto bc_tag = bam_aux_get(record, "BC"); bc_tag != nullptr) {
            alias = bam_aux2Z(bc_tag);
        }

        if (const auto rg_tag = bam_aux_get(record, "RG"); rg_tag != nullptr) {
            std::string rg_tag_value = bam_aux2Z(rg_tag);
            if (auto index = rg_tag_value.find(alias); index != rg_tag_value.npos && index != 0) {
                read_group = rg_tag_value.substr(0, index - 1);
            }
        }
    }

    // Lookup the ReadAttributes for this read_group
    const auto attr_it = m_read_group_to_attributes.find(read_group);
    if (attr_it == m_read_group_to_attributes.cend()) {
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
