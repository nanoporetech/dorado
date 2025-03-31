#include "vcf_writer.h"

#include <htslib/vcf.h>

#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dorado::secondary {

// RAII for the BCF header.
void BcfHdrDestructor::operator()(bcf_hdr_t* p) {
    if (p) {
        bcf_hdr_destroy(p);
    }
}

// RAII for a single BCF record.
struct BcfRecordDestructor {
    void operator()(bcf1_t*);
};
void BcfRecordDestructor::operator()(bcf1_t* p) {
    if (p) {
        bcf_destroy(p);
    }
}
using BcfRecordPtr = std::unique_ptr<bcf1_t, BcfRecordDestructor>;

VCFWriter::VCFWriter(const std::filesystem::path& in_fn,
                     const std::vector<std::pair<std::string, std::string>>& filters,
                     const std::vector<std::pair<std::string, int64_t>>& contigs)
        : m_vcf_fp{hts_open(in_fn.string().c_str(), "w"), HtsFileDestructor()},
          m_header{bcf_hdr_init("w"), BcfHdrDestructor()} {
    if (!m_vcf_fp) {
        throw std::runtime_error("Failed to open VCF file: " + in_fn.string());
    }
    if (!m_header) {
        throw std::runtime_error("Failed to create VCF header.");
    }

    // Add the VCF format version
    bcf_hdr_append(m_header.get(), "##fileformat=VCFv4.1");

    // Add contig information
    for (const auto& [name, length] : contigs) {
        const std::string contig_entry =
                "##contig=<ID=" + name + ",length=" + std::to_string(length) + ">";
        bcf_hdr_append(m_header.get(), contig_entry.c_str());
    }

    // Add FILTER entries.
    for (const auto& [id, description] : filters) {
        const std::string filter_entry =
                "##FILTER=<ID=" + id + ",Description=\"" + description + "\">";  // NOLINT
        bcf_hdr_append(m_header.get(), filter_entry.c_str());
    }

    // Add mandatory INFO and FORMAT fields
    bcf_hdr_append(m_header.get(), "##dorado_version=");
    bcf_hdr_append(m_header.get(),
                   "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">");
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality score\">");

    if (bcf_hdr_add_sample(m_header.get(), "SAMPLE") != 0) {
        throw std::runtime_error("Failed to add sample: SAMPLE");
    }

    // Add column headers
    bcf_hdr_append(m_header.get(), "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE");

    // Write the header to the file
    if (bcf_hdr_write(m_vcf_fp.get(), m_header.get()) < 0) {
        throw std::runtime_error("Failed to write VCF header.");
    }
}

void VCFWriter::write_variant(const Variant& variant) {
    BcfRecordPtr record{bcf_init(), BcfRecordDestructor()};

    if (!record) {
        throw std::runtime_error("Failed to create VCF record.");
    }

    // Format the alleles for Bcftools.
    std::ostringstream os_alleles;
    os_alleles << variant.ref;
    for (const std::string_view alt : variant.alts) {
        os_alleles << ',' << alt;
    }

    // Set the record fields.
    record->rid = variant.seq_id;
    record->pos = variant.pos;
    bcf_update_id(m_header.get(), record.get(), ".");
    bcf_update_alleles_str(m_header.get(), record.get(), os_alleles.str().c_str());
    record->qual = variant.qual;

    // Look up the FILTER ID in the header
    if (!std::empty(variant.filter)) {
        int32_t filter_id = bcf_hdr_id2int(m_header.get(), BCF_DT_ID, variant.filter.c_str());
        if (filter_id < 0) {
            throw std::runtime_error("VCF filter ID '" + variant.filter + "' not found in header.");
        }
        bcf_update_filter(m_header.get(), record.get(), &filter_id, 1);
    }

    // Add INFO fields.
    for (const auto& [key, value] : variant.info) {
        bcf_update_info_string(m_header.get(), record.get(), key.c_str(), value.c_str());
    }

    // Genotype.
    {
        std::vector<std::string> format_keys;
        std::vector<int32_t> format_values;
        for (const auto& [key, value] : variant.genotype) {
            if (key == "GT") {
                format_keys.emplace_back(key);
                format_values.emplace_back(bcf_gt_unphased(value));
                break;
            }
        }
        for (const auto& [key, value] : variant.genotype) {
            if (key != "GT") {
                format_keys.emplace_back(key);
                format_values.emplace_back(value);
            }
        }
        if (std::empty(format_keys) || (format_keys.front() != "GT")) {
            throw std::runtime_error("Genotype key GT not found in variant.genotype!");
        }
        bcf_update_genotypes(m_header.get(), record.get(), &format_values[0], 1);
        for (size_t i = 1; i < std::size(format_keys); ++i) {
            bcf_update_format_int32(m_header.get(), record.get(), format_keys[i].c_str(),
                                    &format_values[i], 1);
        }
    }

    // Write the record.
    if (bcf_write(m_vcf_fp.get(), m_header.get(), record.get()) < 0) {
        throw std::runtime_error("Failed to write VCF record.");
    }
}

}  // namespace dorado::secondary
