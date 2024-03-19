#include "parse_custom_sequences.h"

#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"

#include <htslib/sam.h>
#include <toml.hpp>
#include <toml/value.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace dorado::demux {

std::unordered_map<std::string, std::string> parse_custom_sequences(
        const std::string& sequences_file) {
    auto file = hts_open(sequences_file.c_str(), "r");
    BamPtr record;
    record.reset(bam_init1());

    std::unordered_map<std::string, std::string> sequences;

    int sam_ret_val = 0;
    while ((sam_ret_val = sam_read1(file, nullptr, record.get())) != -1) {
        if (sam_ret_val < -1) {
            throw std::runtime_error("Failed to parse custom sequence file " + sequences_file);
        }
        std::string qname = bam_get_qname(record.get());
        std::string seq = utils::extract_sequence(record.get());
        sequences[qname] = seq;
    }

    hts_close(file);

    return sequences;
}

}  // namespace dorado::demux
