#include "secondary/common/vcf_writer.h"

#include "dorado_version.h"

#include <htslib/hts.h>
#include <htslib/vcf.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dorado::secondary {

namespace {

void ensure_buffer_initialized(kstring_t& buffer) {
    if (buffer.s != nullptr) {
        return;
    }

    if (hts_resize(char, 1, &buffer.m, &buffer.s, HTS_RESIZE_CLEAR) < 0) {
        throw std::runtime_error("Failed to allocate a BCF record buffer.");
    }
    buffer.l = 0;
}

void ensure_record_buffers_initialized(bcf1_t& record) {
    // Workaround for a Htslib ASAN/UBSAN bug.
    // Htslib's allele update path computes rlen via pointer arithmetic on shared.s and,
    // for gVCF alleles with a LEN header present, indiv.s. Under ASAN/UBSAN, a null
    // buffer trips that path before any record data has been synced into the block. Use
    // Htslib's resize helper so the buffers are allocated and later freed on the same
    // side of the library boundary.
    ensure_buffer_initialized(record.shared);
    ensure_buffer_initialized(record.indiv);
}

float missing_bcf_float() {
    static_assert(sizeof(float) == sizeof(std::uint32_t));

    // Same bit pattern as htslib's bcf_float_missing, without referencing the
    // exported data symbol that is not available in all Windows builds.
    //
    // It looks strange - but it matches the original implementation in vcf.c/.h.
    constexpr std::uint32_t missing_value = 0x7F800001;
    float value = 0.0f;
    std::memcpy(&value, &missing_value, sizeof(value));
    return value;
}

}  // namespace

VCFWriter::VCFWriter(const std::filesystem::path& in_fn,
                     const std::vector<std::pair<std::string, std::string>>& filters,
                     const std::vector<std::pair<std::string, int64_t>>& contigs,
                     const bool include_gvcf_headers)
        : m_vcf_fp{hts_open(in_fn.string().c_str(), "w"), HtsFileDestructor()},
          m_header{bcf_hdr_init("w"), BcfHdrDestructor()},
          m_include_gvcf_headers{include_gvcf_headers} {
    if (!m_vcf_fp) {
        throw std::runtime_error("Failed to open VCF file: " + in_fn.string());
    }
    if (!m_header) {
        throw std::runtime_error("Failed to create VCF header.");
    }

    // Set the VCF format version
    bcf_hdr_set_version(m_header.get(), "VCFv4.2");

    // Add FILTER entries.
    for (const auto& [id, description] : filters) {
        const auto filter_entry = std::string("##FILTER=<ID=")
                                          .append(id)
                                          .append(",Description=\"")
                                          .append(description)
                                          .append("\">");
        bcf_hdr_append(m_header.get(), filter_entry.c_str());
    }

    // Add contig information
    for (const auto& [name, length] : contigs) {
        const std::string contig_entry =
                "##contig=<ID=" + name + ",length=" + std::to_string(length) + ">";
        bcf_hdr_append(m_header.get(), contig_entry.c_str());
    }

    // Add mandatory INFO and FORMAT fields
    bcf_hdr_append(m_header.get(), ("##dorado_version=" + std::string(DORADO_VERSION)).c_str());
    bcf_hdr_append(m_header.get(),
                   "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">");
    if (include_gvcf_headers) {
        bcf_hdr_append(m_header.get(),
                       "##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the "
                       "reference block\">");
        bcf_hdr_append(m_header.get(),
                       "##FORMAT=<ID=LEN,Number=1,Type=Integer,Description=\"Length of <*> "
                       "reference block\">");
    }
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality score\">");
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">");
    bcf_hdr_append(m_header.get(),
                   "##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description=\"Minimum read depth\">");

    if (bcf_hdr_add_sample(m_header.get(), "SAMPLE") != 0) {
        throw std::runtime_error("Failed to add sample: SAMPLE");
    }

    // Write the header to the file
    if (bcf_hdr_write(m_vcf_fp.get(), m_header.get()) < 0) {
        throw std::runtime_error("Failed to write VCF header.");
    }
}

void VCFWriter::write_variant(const Variant& variant) {
    BcfRecordPtr record{bcf_init(), BcfRecordDestructor{}};

    if (!record) {
        throw std::runtime_error("Failed to create VCF record.");
    }

    ensure_record_buffers_initialized(*record);

    std::vector<std::string> alts = variant.alts;

    // Add "<*>" to non-reference records in gVCF output.
    if (m_include_gvcf_headers) {
        if (!is_reference_record(variant) &&
            (std::find(std::cbegin(alts), std::cend(alts), "<*>") == std::cend(alts))) {
            alts.emplace_back("<*>");
        }
    }

    // Format the alleles for Bcftools.
    std::ostringstream os_alleles;
    os_alleles << variant.ref;
    for (const std::string_view alt : alts) {
        os_alleles << ',' << alt;
    }

    // Set the record fields.
    record->rid = variant.seq_id;
    record->pos = variant.pos;
    bcf_update_id(m_header.get(), record.get(), ".");
    bcf_update_alleles_str(m_header.get(), record.get(), os_alleles.str().c_str());
    if (variant.qual < 0.0f) {
        record->qual = missing_bcf_float();
    } else {
        record->qual = variant.qual;
    }

    // Look up the FILTER ID in the header
    if (!std::empty(variant.filter) && (variant.filter != ".")) {
        int32_t filter_id = bcf_hdr_id2int(m_header.get(), BCF_DT_ID, variant.filter.c_str());
        if (filter_id < 0) {
            throw std::runtime_error("VCF filter ID '" + variant.filter + "' not found in header.");
        }
        bcf_update_filter(m_header.get(), record.get(), &filter_id, 1);
    }

    // Add INFO fields.
    for (const auto& [key, value] : variant.info) {
        if (key == "END") {
            const int32_t end = std::stoi(value);
            bcf_update_info_int32(m_header.get(), record.get(), key.c_str(), &end, 1);
        } else {
            bcf_update_info_string(m_header.get(), record.get(), key.c_str(), value.c_str());
        }
    }

    // Genotype.
    {
        std::vector<std::string> format_keys;
        std::vector<int32_t> format_values;
        std::vector<int32_t> genotype_values;

        for (const auto& [key, value] : variant.genotype) {
            if (key == "GT") {
                std::istringstream ss(value);
                std::string token;
                while (std::getline(ss, token, '/')) {
                    if (token == ".") {
                        genotype_values.emplace_back(bcf_gt_missing);
                    } else {
                        genotype_values.emplace_back(bcf_gt_unphased(std::stoi(token)));
                    }
                }
            } else {
                format_keys.emplace_back(key);
                format_values.emplace_back((value == ".") ? bcf_int32_missing : std::stoi(value));
            }
        }

        if (std::empty(genotype_values)) {
            throw std::runtime_error("No genotype information found in variant.genotype!");
        }

        // Update the genotype.
        bcf_update_genotypes(m_header.get(), record.get(), std::data(genotype_values),
                             std::size(genotype_values));

        // Update other keys (like genotype quality).
        for (int64_t i = 0; i < std::ssize(format_keys); ++i) {
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
