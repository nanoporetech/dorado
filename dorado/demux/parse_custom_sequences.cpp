#include "parse_custom_sequences.h"

#include "utils/bam_utils.h"
#include "utils/types.h"

#include <htslib/sam.h>

#include <stdexcept>

namespace dorado::demux {

std::unordered_map<std::string, std::string> parse_custom_sequences(
        const std::string& sequences_file) {
    dorado::HtsFilePtr file(hts_open(sequences_file.c_str(), "r"));
    if (!file) {
        throw std::runtime_error("Unable to open file " + sequences_file);
    }

    BamPtr record{bam_init1()};

    std::unordered_map<std::string, std::string> sequences;

    int sam_ret_val = 0;
    while ((sam_ret_val = sam_read1(file.get(), nullptr, record.get())) != -1) {
        if (sam_ret_val < -1) {
            throw std::runtime_error("Failed to parse custom sequence file " + sequences_file);
        }
        std::string qname = bam_get_qname(record.get());
        std::string seq = utils::extract_sequence(record.get());
        sequences[qname] = seq;
    }

    return sequences;
}

}  // namespace dorado::demux
