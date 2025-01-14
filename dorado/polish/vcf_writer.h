#pragma once

#include "variant.h"

#include <htslib/vcf.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dorado::polisher {

// class VCFWriter {
// public:
//     VCFWriter();
// };

class VCFWriter {
private:
    htsFile* vcf_file_ = nullptr;
    bcf_hdr_t* header_ = nullptr;

public:
    VCFWriter(const std::string& filename,
              const std::vector<std::pair<std::string, std::string>>& filters,
              const std::vector<std::pair<std::string, int64_t>>& contigs);

    ~VCFWriter();

    void write_variant(const Variant& variant);
};

// void write_vcf(const std::string& filename, const std::vector<std::pair<std::string, int64_t>>& contigs

}  // namespace dorado::polisher
