#pragma once

#include "hts_utils/MergeHeaders.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_types.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

struct sam_hdr_t;
struct bam1_t;

namespace dorado::utils {

class HeaderMapper {
public:
    // A function for modifying all finalised headers.
    using Modifier = std::function<void(sam_hdr_t* hdr)>;
    // Mapping of read attributes to merged headers
    using HeaderMap = std::unordered_map<HtsData::ReadAttributes,
                                         std::unique_ptr<MergeHeaders>,
                                         HtsData::ReadAttributesCoreHasher,
                                         HtsData::ReadAttributesCoreComparator>;

    // Mapping of read group ID to read attributes;
    using AttributeMap = std::unordered_map<std::string, HtsData::ReadAttributes>;

    /** Construct a mapping of structured output key to merged headers 
     *  @param inputs Collection of input file hts file inputs to map
     *  @param strip_alignment If set, no SQ lines will be included in the
     *         merged header, and no checks will be made for SQ conflicts. 
     */
    HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment);

    // Apply a HeaderModifier function to all merged headers.
    void modify_headers(const Modifier& modifier) const;

    std::shared_ptr<HeaderMap> get_merged_headers_map() const { return m_merged_headers_map; };
    std::shared_ptr<AttributeMap> get_read_attributes_map() const {
        return m_read_group_to_attributes;
    }

    const HtsData::ReadAttributes& get_read_attributes(const bam1_t* record) const;
    const MergeHeaders& get_merged_header(const HtsData::ReadAttributes& attrs) const;

    // Create a shared sam header by merging all dynamic headers.
    SamHdrPtr get_shared_merged_header(bool strip_alignments) const;

private:
    const bool m_strip_alignment;
    // Store this fallback for when we don't find a read group
    // This can happen when a FASTQ has no header metadata
    HtsData::ReadAttributes m_fallback_read_attrs{};

    std::shared_ptr<AttributeMap> m_read_group_to_attributes{nullptr};
    std::shared_ptr<HeaderMap> m_merged_headers_map{nullptr};

    void process(const std::vector<std::filesystem::path>& inputs);
    void process_bam(const std::filesystem::path& path);
    void process_fastx(const std::filesystem::path& path);

    AttributeMap get_read_attrs_by_id(const std::vector<utils::HeaderLineData>& header_lines);

    bool m_fastq_runtime_warning_issued{false};
};

}  // namespace dorado::utils
