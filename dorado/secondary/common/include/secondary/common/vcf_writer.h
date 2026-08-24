#pragma once

#include "hts_utils/hts_types.h"
#include "variant.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::secondary {

class VCFWriter {
public:
    /**
     * \brief Opens a VCF file for writing.
     * \param filters A vector of all possible filters which can appear in this VCF. Required, or Htslib will fail to produce a record. Pair is: <filter_name, description>. The description is a free-text description of the filter.
     * \param contigs A vector of header/length pairs for every contig which may appear in this VCF.
     */
    VCFWriter(const std::filesystem::path& filename,
              const std::vector<std::pair<std::string_view, std::string>>& filters,
              const std::vector<std::pair<std::string, int64_t>>& contigs,
              bool include_gvcf_headers);

    void write_variant(const Variant& variant);

private:
    HtsFilePtr m_vcf_fp;
    BcfHdrPtr m_header;
    bool m_include_gvcf_headers = false;
};

}  // namespace dorado::secondary
