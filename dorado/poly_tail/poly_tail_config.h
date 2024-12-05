#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace dorado::poly_tail {

struct PolyTailConfig {
    std::string rna_adapter = "GGTTGTTTCTGTTGGTGCTG";                                // RNA
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
    int min_primer_separation = 10;
    float flank_threshold = 0.6f;
    bool is_plasmid = false;
    int tail_interrupt_length = 0;
    int min_base_count = 10;
    std::string barcode_id;
};

// Prepare the PolyA configurations. If a configuration file is available, parse it to extract parameters.
// If barcode-specific overrides are present, the non-specific configuration will be at the back.
// Otherwise prepares a single default configuration.
std::vector<PolyTailConfig> prepare_configs(const std::string& config_file);

// Overloaded function that parses the configuration passed
// in as an input stream.
std::vector<PolyTailConfig> prepare_configs(std::istream& is);

}  // namespace dorado::poly_tail
