#pragma once

#include "utils/types.h"
#include "variant.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

struct bcf_hdr_t;

namespace dorado::secondary {

// RAII for the BCF header.
struct BcfHdrDestructor {
    void operator()(bcf_hdr_t*);
};
using BcfHdrPtr = std::unique_ptr<bcf_hdr_t, BcfHdrDestructor>;

class VCFWriter {
public:
    /**
     * \brief Opens a VCF file for writing.
     * \param filters A vector of all possible filters which can appear in this VCF. Required, or Htslib will fail to produce a record. Pair is: <filter_name, description>. The description is a free-text description of the filter.
     * \param contigs A vector of header/length pairs for every contig which may appear in this VCF.
     */
    VCFWriter(const std::filesystem::path& filename,
              const std::vector<std::pair<std::string, std::string>>& filters,
              const std::vector<std::pair<std::string, int64_t>>& contigs);

    ~VCFWriter() = default;

    void write_variant(const Variant& variant);

private:
    HtsFilePtr m_vcf_fp;
    BcfHdrPtr m_header;
};

}  // namespace dorado::secondary