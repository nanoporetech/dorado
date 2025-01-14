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

    if (!variant.qual.empty()) {
        record->qual = std::stof(variant.qual);
    } else {
        record->qual = bcf_float_missing;
    }

    // Look up the FILTER ID in the header
    if (!variant.filter.empty()) {
        int32_t filter_id = bcf_hdr_id2int(header_, BCF_DT_ID, variant.filter.c_str());
        if (filter_id < 0) {
            throw std::runtime_error("FILTER ID '" + variant.filter + "' not found in header.");
        }
        bcf_update_filter(header_, record, &filter_id, 1);
    }

    // int32_t tmpi = bcf_hdr_id2int(hdr, BCF_DT_ID, "PASS");
    // bcf_update_filter(header_, record, variant.filter.empty() ? NULL : variant.filter.c_str(), variant.filter.empty() ? 0 : 1);
    // // bcf_update_filter(header_, record, variant.filter.empty() ? NULL : variant.filter.c_str(), variant.filter.empty() ? 0 : 1);

    // Add INFO fields
    for (const auto& [key, value] : variant.info) {
        bcf_update_info_string(header_, record, key.c_str(), value.c_str());
    }

    // // Add genotype information. Make sure that the GT tag is first.
    // std::vector<std::pair<std::string, std::string>> genotype;
    // for (const auto& [key, value] : variant.genotype) {
    //     if (key == "GT") {
    //         genotype.emplace_back(key, value);
    //         break;
    //     }
    // }
    // for (const auto& [key, value] : variant.genotype) {
    //     if (key != "GT") {
    //         genotype.emplace_back(key, value);
    //     }
    // }
    // for (const auto& [key, value] : genotype) {
    //     bcf_update_genotypes(header_, record, value.c_str(), 1);
    // }

    // Process genotype to ensure GT comes first
    {
        // // Update genotype
        // std::string format = "GT:GQ";
        // std::string sample = variant.genotype[0].second + ":" + variant.genotype[1].second;
        // bcf_update_format(header_, record, "GT", &variant.genotype[0].second, 1, BCF_HT_STR);
        // int gq = std::stoi(variant.genotype[1].second);
        // bcf_update_format_int32(header_, record, "GQ", &gq, 1);

        // // Construct FORMAT and SAMPLE fields
        // std::vector<std::string> format_keys;
        // std::vector<int> gt_values; // For storing GT as integers
        // std::vector<float> gq_values; // For storing GQ as floats

        // for (const auto& [key, value] : variant.genotype) {
        //     if (key == "GT") {
        //         format_keys.push_back("GT");
        //         // Convert GT string (e.g., "1") to integer
        //         gt_values.push_back(std::stoi(value));
        //     } else if (key == "GQ") {
        //         format_keys.push_back("GQ");
        //         // Convert GQ string (e.g., "26") to float
        //         gq_values.push_back(std::stof(value));
        //     }
        // }

        // // Create the FORMAT string
        // std::string format = format_keys[0];
        // for (size_t i = 1; i < format_keys.size(); ++i) {
        //     format += ":" + format_keys[i];
        // }
        // std::array<const char*, 1> format_arr{format.c_str()};

        // // Update FORMAT field
        // bcf_update_format_string(header_, record, "FORMAT", format_arr.data(), 1);

        // // Update SAMPLE fields
        // if (!gt_values.empty()) {
        //     bcf_update_format_int32(header_, record, "GT", gt_values.data(), gt_values.size());
        // }
        // if (!gq_values.empty()) {
        //     bcf_update_format_int32(header_, record, "GQ", gq_values.data(), gq_values.size());
        // }

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
        // const std::string gt_str = std::to_string(format_values[0]);
        // std::cerr << "gt_str = " << gt_str << "\n";
        // std::array<const char*, 1> gt_str_arr{gt_str.c_str()};
        // bcf_update_genotypes(header_, record, &gt_str_arr[0], 1);
        bcf_update_genotypes(header_, record, &format_values[0], 1);
        for (size_t i = 1; i < std::size(format_keys); ++i) {
            bcf_update_format_int32(header_, record, format_keys[i].c_str(), &format_values[i], 1);
        }

        // bcf_update_format_string(header_, record, format.c_str(), vals_arr.data(), vals_arr.size());

        // std::string format;
        // // std::string vals;
        // // std::vector<std::string> format_keys;
        // // std::vector<int32_t> format_values;
        // std::vector<std::string> format_values_str;
        // for (const auto& [key, value] : variant.genotype) {
        //     if (key == "GT") {
        //         format = "GT";
        //         // vals = std::to_string(value);
        //         // format_keys.emplace_back(key);
        //         // format_values.emplace_back(value);
        //         format_values_str.emplace_back(std::to_string(value));
        //         break;
        //     }
        // }
        // for (const auto& [key, value] : variant.genotype) {
        //     if (key != "GT") {
        //         format += ":" + key;
        //         // vals += ":" + std::to_string(value);
        //         // format_keys.emplace_back(key);
        //         // format_values.emplace_back(value);
        //         format_values_str.emplace_back(std::to_string(value));
        //     }
        // }
        // // Convert format values to const char* array
        // std::vector<const char*> vals_arr;
        // for (const auto& v : format_values_str) {
        //     vals_arr.emplace_back(v.c_str());
        // }

        // // bcf_update_format_string(header_, record, format.c_str(), vals_arr.data(), vals_arr.size());

        // // Construct FORMAT and SAMPLE fields
        // std::string format = "GT"; // Start with GT
        // std::vector<int32_t> gt_values; // Genotype values
        // std::vector<int32_t> other_values; // Values for other FORMAT keys
        // std::vector<std::string> other_keys; // Other FORMAT keys

        // // Extract GT first
        // for (const auto& [key, value] : variant.genotype) {
        //     if (key == "GT") {
        //         gt_values.push_back(bcf_gt_unphased(value));
        //     } else {
        //         other_keys.push_back(key);
        //         other_values.push_back(value);
        //     }
        // }

        // // Update FORMAT fields
        // int n_samples = bcf_hdr_nsamples(header_);
        // if (n_samples == 0) {
        //     throw std::runtime_error("No samples defined in VCF header.");
        // }

        // // Resize arrays to match the number of samples
        // gt_values.resize(n_samples, bcf_int32_missing);
        // other_values.resize(n_samples * other_keys.size(), bcf_int32_missing);

        // // Update GT
        // bcf_update_genotypes(header_, record, gt_values.data(), n_samples);

        // // // Update other FORMAT fields
        // // for (size_t i = 0; i < other_keys.size(); ++i) {
        // //     bcf_update_format_int32(
        // //         header_, record, other_keys[i].c_str(),
        // //         &other_values[i * n_samples], n_samples);
        // // }

        // std::cerr << "format = " << format << ", vals = " << vals << "\n";
        // bcf_update_format_string(header_, record, format_keys.front().c_str(), vals_arr.data(), vals_arr.size());

        // bcf_update_genotypes(header_, record, &format_values[0], 1);
        // for (size_t i = 1; i < std::size(format_keys); ++i) {
        //     bcf_update_format_int32(header_, record, format_keys[i].c_str(), &format_values[i], 1);
        // }

        // // Update GT first
        // std::vector<int32_t> gt_values;
        // for (const auto& [key, value] : variant.genotype) {
        //     if (key == "GT") {
        //         gt_values.push_back(value);  // Convert GT string to int
        //         break;
        //     }
        // }
        // if (!gt_values.empty()) {
        //     bcf_update_genotypes(header_, record, gt_values.data(), gt_values.size());
        // }

        // // Update other FORMAT fields
        // for (const auto& [key, value] : variant.genotype) {
        //     if (key != "GT") {
        //         bcf_update_format_int32(header_, record, key.c_str(), &value, 1);
        //     }
        // }

        // // Convert format values to const char* array
        // std::vector<const char*> format_keys_arr;
        // for (const auto& key : format_keys) {
        //     format_keys_arr.push_back(key.c_str());
        // }

        // // std::vector<const int32_t*> format_values_arr;
        // // for (const auto& value : format_values) {
        // //     format_values_arr.push_back(value.c_str());
        // // }
        // for (size_t i = 0; i < std::size(format_keys); ++i) {
        //     std::cerr << "[i = " << i << "], format_keys = " << format_keys[i] << ", format_values[i] = " << format_values[i] << "\n";
        // }

        // const std::string gt_val = std::to_string(format_values.front());
        // std::array<const char*, 1> gt_val_array{gt_val.c_str()};
        // (void) gt_val_array;
        // // bcf_update_format_string(header_, record, format_keys[0].c_str(), gt_val_array.data(), 1);

        // bcf_update_genotypes(header_, record, &format_values[0], 1);
        // for (size_t i = 1; i < std::size(format_keys); ++i) {
        //     bcf_update_format_int32(header_, record, format_keys[i].c_str(), &format_values[i], 1);
        // }

        // int gq = std::stoi(variant.genotype[1].second);
        // bcf_update_format_int32(header_, record, "GQ", &gq, 1);

        // std::string format = format_keys.front();
        // for (size_t i = 1; i < format_keys.size(); ++i) {
        //     format += ":" + format_keys[i];
        // }
        // const std::array<const char*, 1> format_cstr{format.c_str()};

        // // Update FORMAT field
        // bcf_update_format_string(header_, record, "FORMAT", format_cstr.data(), 1);

        // // bcf_update_format_string(header_, record, "FORMAT", format_keys_cstr.data(), format_keys_cstr.size());
        // // bcf_update_format_string(header_, record, "SAMPLE", format_values_cstr.data(), format_values_cstr.size());

        // std::string format;
        // std::vector<std::string> sample_values;

        // for (const auto& [key, value] : variant.genotype) {
        //     if (key == "GT") {
        //         format = "GT";
        //         sample_values.push_back(value);
        //     }
        // }

        // if (std::empty(format)) {
        //     throw std::runtime_error("Could not find the GT key in variant.genotype.");
        // }

        // for (const auto& [key, value] : variant.genotype) {
        //     if (key != "GT") {
        //         format += ":" + key;
        //         sample_values.push_back(value);
        //     }
        // }

        // // Update FORMAT and SAMPLE fields.
        // bcf_update_format_string(header_, record, format.c_str(), format.size());
        // bcf_update_format(header_, record, "GT", sample_values.data(), sample_values.size(), BCF_HT_STR);
    }

    // Write the record to the file
    if (bcf_write(vcf_file_, header_, record) < 0) {
        bcf_destroy(record);
        throw std::runtime_error("Failed to write VCF record.");
    }

    // Free the record
    bcf_destroy(record);
}

}  // namespace dorado::polisher
