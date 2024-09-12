#pragma once

#include <istream>
#include <string>
#include <vector>

namespace dorado::poly_tail {

struct PolyTailConfig {
    std::string rna_adapter = "GGTTGTTTCTGTTGGTGCTGATATTGC";                         // RNA
    std::string front_primer = "TTTCTGTTGGTGCTGATATTGCTTT";                          // SSP
    std::string rear_primer = "ACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTTT";  // VNP
    std::string rc_front_primer;
    std::string rc_rear_primer;
    std::string plasmid_front_flank;
    std::string plasmid_rear_flank;
    std::string rc_plasmid_front_flank;
    std::string rc_plasmid_rear_flank;
    int rna_offset = 61;
    int primer_window = 150;
    float flank_threshold = 0.6f;
    bool is_plasmid = false;
    int tail_interrupt_length = 0;
    int min_base_count = 10;
    std::string barcode_id;
};

// Prepare the PolyA configuration struct. If a configuration
// file is available, parse it to extract parameters. Otherwise
// prepare the default configuration.
std::vector<PolyTailConfig> prepare_configs(const std::string& config_file);

// Overloaded function that parses the configuration passed
// in as an input stream.
std::vector<PolyTailConfig> prepare_configs(std::istream& is);

}  // namespace dorado::poly_tail
