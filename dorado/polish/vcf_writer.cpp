#include "vcf_writer.h"

namespace dorado::polisher {

// VCFWriter::VCFWriter() {

// }

VCFWriter::VCFWriter(const std::string& filename,
                     const std::vector<std::pair<std::string, std::string>>& filters,
                     const std::vector<std::pair<std::string, int64_t>>& contigs) {
    // Open the file for writing
    vcf_file_ = hts_open(filename.c_str(), "w");
    if (!vcf_file_) {
        throw std::runtime_error("Failed to open VCF file: " + filename);
    }

    // Create a new VCF header
    header_ = bcf_hdr_init("w");
    if (!header_) {
        throw std::runtime_error("Failed to create VCF header.");
    }

    // Add the VCF format version
    bcf_hdr_append(header_, "##fileformat=VCFv4.1");

    // Add contig information
    for (const auto& [name, length] : contigs) {
        std::string contig_entry =
                "##contig=<ID=" + name + ",length=" + std::to_string(length) + ">";
        bcf_hdr_append(header_, contig_entry.c_str());
    }

    // Add FILTER entries.
    for (const auto& [id, description] : filters) {
        std::string filter_entry = "##FILTER=<ID=" + id + ",Description=\"" + description + "\">";
        bcf_hdr_append(header_, filter_entry.c_str());
    }

    // Add mandatory INFO and FORMAT fields
    bcf_hdr_append(header_, "##dorado_version=");
    bcf_hdr_append(header_, "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">");
    bcf_hdr_append(header_, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    bcf_hdr_append(header_,
                   "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality score\">");

    if (bcf_hdr_add_sample(header_, "SAMPLE") != 0) {
        throw std::runtime_error("Failed to add sample: SAMPLE");
    }

    // Add column headers
    bcf_hdr_append(header_, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE");

    // Write the header to the file
    if (bcf_hdr_write(vcf_file_, header_) < 0) {
        throw std::runtime_error("Failed to write VCF header.");
    }
}

VCFWriter::~VCFWriter() {
    if (header_) {
        bcf_hdr_destroy(header_);
    }
    if (vcf_file_) {
        hts_close(vcf_file_);
    }
}

void VCFWriter::write_variant(const Variant& variant) {
    // Create a new VCF record
    bcf1_t* record = bcf_init();
    if (!record) {
        throw std::runtime_error("Failed to create VCF record.");
    }

    // Set the record fields
    record->rid = variant.seq_id;
    record->pos = variant.pos;
    bcf_update_id(header_, record, ".");
    bcf_update_alleles_str(header_, record, (variant.ref + "," + variant.alt).c_str());

    record->qual = variant.qual;

    // Look up the FILTER ID in the header
    if (!variant.filter.empty()) {
        int32_t filter_id = bcf_hdr_id2int(header_, BCF_DT_ID, variant.filter.c_str());
        if (filter_id < 0) {
            throw std::runtime_error("FILTER ID '" + variant.filter + "' not found in header.");
        }
        bcf_update_filter(header_, record, &filter_id, 1);
    }

    // Add INFO fields
    for (const auto& [key, value] : variant.info) {
        bcf_update_info_string(header_, record, key.c_str(), value.c_str());
    }

    // Process genotype to ensure GT comes first
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
        bcf_update_genotypes(header_, record, &format_values[0], 1);
        for (size_t i = 1; i < std::size(format_keys); ++i) {
            bcf_update_format_int32(header_, record, format_keys[i].c_str(), &format_values[i], 1);
        }
    }

    // Write the record.
    if (bcf_write(vcf_file_, header_, record) < 0) {
        bcf_destroy(record);
        throw std::runtime_error("Failed to write VCF record.");
    }

    // Free the record
    bcf_destroy(record);
}

}  // namespace dorado::polisher
