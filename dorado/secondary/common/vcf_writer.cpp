#include "secondary/common/vcf_writer.h"

#include "dorado_version.h"

#include <htslib/hts.h>
#include <htslib/vcf.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dorado::secondary {

namespace {

enum class FieldType { GENOTYPE, INT_SCALAR, INT_LIST, STRING };

struct FieldSpec {
    std::string_view key;
    FieldType type;
    std::string_view number;
    std::string_view description;
};

constexpr std::array FORMAT_FIELD_SPECS{
        FieldSpec{"GT", FieldType::GENOTYPE, "1", "Genotype"},
        FieldSpec{"GQ", FieldType::INT_SCALAR, "1", "Genotype quality score"},
        FieldSpec{"LEN", FieldType::INT_SCALAR, "1", "Length of <*> reference block"},
        FieldSpec{"DP", FieldType::INT_SCALAR, "1", "Read depth"},
        FieldSpec{"MIN_DP", FieldType::INT_SCALAR, "1", "Minimum read depth"},
        FieldSpec{"SRC", FieldType::STRING, "1", "Source of the variant"},
        // FieldSpec{"AD", FieldType::INT_LIST, "R", "Allele read depth"},
};

constexpr std::array INFO_FIELD_SPECS{
        FieldSpec{"END", FieldType::INT_SCALAR, "1", "End position of the reference block"},
};

std::string_view format_field_type(const FieldType& type) {
    return ((type == FieldType::GENOTYPE) || (type == FieldType::STRING)) ? "String" : "Integer";
}

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

int32_t parse_vcf_int(const std::string_view value) {
    if (value == ".") {
        return bcf_int32_missing;
    }
    return utils::from_chars<int32_t>(value).value();
}

std::vector<int32_t> parse_vcf_int_list(const std::string_view value) {
    const auto tokens = utils::split_view(value, ',');
    std::vector<int32_t> ret;
    ret.reserve(std::size(tokens));
    for (const auto token : tokens) {
        ret.push_back(parse_vcf_int(token));
    }
    if (ret.empty()) {
        throw std::runtime_error("FORMAT integer list is empty.");
    }
    return ret;
}

std::string format_header_line(const FieldSpec& spec, const std::string_view meta_key) {
    return std::string("##")
            .append(meta_key)
            .append("=<ID=")
            .append(spec.key)
            .append(",Number=")
            .append(spec.number)
            .append(",Type=")
            .append(format_field_type(spec.type))
            .append(",Description=\"")
            .append(spec.description)
            .append("\">");
}

}  // namespace

VCFWriter::VCFWriter(const std::filesystem::path& in_fn,
                     const Filters::FilterType& filters,
                     const std::vector<std::pair<std::string, int64_t>>& contigs,
                     const bool include_gvcf_headers,
                     const std::string_view command_line)
        : m_vcf_fp{hts_open(in_fn.string().c_str(), "w")},
          m_header{bcf_hdr_init("w")},
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
    for (size_t n = 0; n < std::size(KNOWN_FILTERS); ++n) {
        Filters::FilterType filtbit = 1 << n;
        if (filtbit & Filters::Pass) {
            // PASS is added automatically by htslib
            continue;
        }
        if (filtbit & filters) {
            const auto filter_entry = KNOWN_FILTERS[n].header_string();
            bcf_hdr_append(m_header.get(), filter_entry.c_str());
        }
    }

    // Add target sequence information.
    for (const auto& [name, length] : contigs) {
        const std::string contig_entry =
                "##contig=<ID=" + name + ",length=" + std::to_string(length) + ">";
        bcf_hdr_append(m_header.get(), contig_entry.c_str());
    }

    // Add the Dorado version.
    bcf_hdr_append(m_header.get(), ("##dorado_version=" + std::string(DORADO_VERSION)).c_str());

    // Add the command line to the header if provided.
    if (!std::empty(command_line)) {
        const std::string line = "##cmdline=" + std::string(command_line);
        bcf_hdr_append(m_header.get(), line.c_str());
    }

    // Add INFO and FORMAT fields.
    for (const auto& spec : INFO_FIELD_SPECS) {
        if (!include_gvcf_headers && (spec.key == "END")) {
            continue;
        }
        const std::string format_entry = format_header_line(spec, "INFO");
        bcf_hdr_append(m_header.get(), format_entry.c_str());
    }
    for (const auto& spec : FORMAT_FIELD_SPECS) {
        if (!include_gvcf_headers && (spec.key == "LEN")) {
            continue;
        }
        const std::string format_entry = format_header_line(spec, "FORMAT");
        bcf_hdr_append(m_header.get(), format_entry.c_str());
    }

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
    if (variant.filters == Filters::Missing) {
        bcf_update_filter(m_header.get(), record.get(), nullptr, 0);
    }
    for (const std::string_view filt : variant.filters.as_strings()) {
        // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage): filter is NUL terminated.
        int32_t filter_id = bcf_hdr_id2int(m_header.get(), BCF_DT_ID, filt.data());
        if (filter_id < 0) {
            throw std::runtime_error("VCF filter ID '" + std::string(filt) +
                                     "' not found in header.");
        }
        bcf_add_filter(m_header.get(), record.get(), filter_id);
    }

    // Add INFO fields.
    {
        for (const FieldSpec& spec : INFO_FIELD_SPECS) {
            const auto it = variant.info.find(std::string(spec.key));
            if (it == variant.info.cend()) {
                continue;
            }
            const std::string& key = it->first;
            const std::string& value = it->second;

            if (spec.type == FieldType::INT_SCALAR) {
                const int32_t end = std::stoi(value);
                bcf_update_info_int32(m_header.get(), record.get(), key.c_str(), &end, 1);
                continue;
            }

            if (spec.type == FieldType::STRING) {
                bcf_update_info_string(m_header.get(), record.get(), key.c_str(), value.c_str());
                continue;
            }

            throw std::runtime_error("Unhandled info field " + std::string(spec.key) +
                                     " with Type " + std::string(format_field_type(spec.type)));
        }
        // Sanity check that the Variant object has no unexpected INFO fields.
        for (const auto& [key, value] : variant.info) {
            if (std::find_if(std::cbegin(INFO_FIELD_SPECS), std::cend(INFO_FIELD_SPECS),
                             [&key](const FieldSpec& spec) { return spec.key == key; }) ==
                std::cend(INFO_FIELD_SPECS)) {
                throw std::runtime_error("Variant object has unexpected INFO field: " + key);
            }
        }
    }

    // Add FORMAT fields.
    {
        std::vector<int32_t> genotype_values;

        for (const FieldSpec& spec : FORMAT_FIELD_SPECS) {
            const auto it = variant.format.find(std::string(spec.key));
            if (it == variant.format.cend()) {
                continue;
            }
            const std::string& value = it->second;

            if (spec.type == FieldType::GENOTYPE) {
                std::istringstream ss(value);
                std::string token;
                while (std::getline(ss, token, '/')) {
                    if (token == ".") {
                        genotype_values.emplace_back(bcf_gt_missing);
                    } else {
                        genotype_values.emplace_back(bcf_gt_unphased(std::stoi(token)));
                    }
                }
                continue;
            }

            if (spec.type == FieldType::INT_SCALAR) {
                const int32_t parsed_value = parse_vcf_int(value);
                bcf_update_format_int32(m_header.get(), record.get(), spec.key.data(),
                                        &parsed_value, 1);
                continue;
            }

            if (spec.type == FieldType::INT_LIST) {
                const auto parsed_values = parse_vcf_int_list(value);
                bcf_update_format_int32(m_header.get(), record.get(), spec.key.data(),
                                        parsed_values.data(), parsed_values.size());
                continue;
            }

            if (spec.type == FieldType::STRING) {
                const char* data = value.c_str();
                // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage): key is NUL terminated.
                bcf_update_format_string(m_header.get(), record.get(), spec.key.data(), &data, 1);
                continue;
            }

            throw std::runtime_error("Unhandled genotype field " + std::string(spec.key) +
                                     " with Type " + std::string(format_field_type(spec.type)));
        }

        // Update the genotype.
        if (std::empty(genotype_values)) {
            throw std::runtime_error("No genotype information found in variant.format!");
        }
        bcf_update_genotypes(m_header.get(), record.get(), std::data(genotype_values),
                             std::size(genotype_values));

        // The variant is missing the SRC FORMAT field; emit the missing value (".").
        if (variant.format.count(FORMAT_TAG_VARIANT_SOURCE) == 0) {
            bcf_update_format_string(m_header.get(), record.get(), FORMAT_TAG_VARIANT_SOURCE,
                                     const_cast<const char**>(&VARIANT_SOURCE_UNKNOWN), 1);
        }

        // Sanity check that the Variant object has no unexpected FORMAT fields.
        for (const auto& [key, value] : variant.format) {
            if (std::find_if(std::cbegin(FORMAT_FIELD_SPECS), std::cend(FORMAT_FIELD_SPECS),
                             [&key](const FieldSpec& spec) { return spec.key == key; }) ==
                std::cend(FORMAT_FIELD_SPECS)) {
                throw std::runtime_error("Variant object has unexpected FORMAT field: " + key);
            }
        }
    }

    // Write the record.
    if (bcf_write(m_vcf_fp.get(), m_header.get(), record.get()) < 0) {
        throw std::runtime_error("Failed to write VCF record.");
    }
}

}  // namespace dorado::secondary
