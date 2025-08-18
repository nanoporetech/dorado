#pragma once

#include "hts_utils/MergeHeaders.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_types.h"

#include <htslib/sam.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace dorado::utils {

class HeaderMapper {
    using HeaderMap = std::unordered_map<HtsData::ReadAttributes,
                                         std::unique_ptr<MergeHeaders>,
                                         HtsData::ReadAttributesCoreHasher,
                                         HtsData::ReadAttributesCoreComparator>;

public:
    /** Construct a mapping of structured output key to merged headers 
     *  @param inputs Collection of input file hts file inputs to map
     *  @param strip_alignment If set, no SQ lines will be included in the
     *         merged header, and no checks will be made for SQ conflicts. 
     */
    HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment);

    const HeaderMap& get_header_map() const { return m_merged_headers_lut; };

protected:
    static std::unordered_map<std::string, HtsData::ReadAttributes> get_read_attrs_lut(
            const std::vector<utils::HeaderLineData>& header_lines);

private:
    const bool m_strip_alignment;

    HeaderMap m_merged_headers_lut;

    void process(const std::vector<std::filesystem::path>& inputs);
    void process_bam(const std::filesystem::path& path);
    void process_fastq(const std::filesystem::path& path);
};

}  // namespace dorado::utils
