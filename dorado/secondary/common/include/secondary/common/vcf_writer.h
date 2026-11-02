#pragma once

#include "hts_utils/hts_types.h"
#include "variant.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::secondary {

extern const std::string VCF_DEFAULT_SAMPLE;

class VCFWriter {
public:
    /**
     * \brief Opens a VCF file for writing.
     * \param filename Output path. A ".gz" extension writes BGZF-compressed output, otherwise writes uncompressed.
                       "-" writes to stdout.
     * \param filters A bitset representation of the allowed filters, which will be read from KNOWN_FILTERS.
     * \param contigs A vector of header/length pairs for every contig which may appear in this VCF.
     * \param include_gvcf_headers Whether gVCF-specific header lines should be added.
     * \param sample_name Name to be used for the sample column.
     * \param command_line The command line to add to the VCF header. An empty value is omitted.
     */
    VCFWriter(const std::filesystem::path& filename,
              const Filters::FilterType filters,
              const std::vector<std::pair<std::string, int64_t>>& contigs,
              bool include_gvcf_headers,
              const std::string& sample_name,
              std::string_view command_line);

    /**
     * \brief Destructor handles flushing and closing the file. If the filename ends in ".gz" the output is BGZF compressed
     *        and a tabix index (<filename>.tbi) is built here.
     */
    ~VCFWriter();

    void write_variant(const Variant& variant);

private:
    std::filesystem::path m_filename;
    bool m_compressed = false;
    HtsFilePtr m_vcf_fp;
    BcfHdrPtr m_header;
    bool m_include_gvcf_headers = false;
};

}  // namespace dorado::secondary
