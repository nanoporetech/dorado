#include "parse_custom_kit.h"

#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"

#include <htslib/sam.h>
#include <toml.hpp>
#include <toml/value.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace dorado::demux {

std::pair<std::string, barcode_kits::KitInfo> parse_custom_arrangement(
        const std::string& arrangement_file) {
    const toml::value config_toml = toml::parse(arrangement_file);

    if (!config_toml.contains("arrangement")) {
        throw std::runtime_error(
                "Custom barcode arrangement file must have [arrangement] section.");
    }

    barcode_kits::KitInfo new_kit;

    const auto& config = toml::find(config_toml, "arrangement");
    std::string kit_name = toml::find<std::string>(config, "name");

    new_kit.name = toml::find<std::string>(config, "kit");

    // Determine barcode sequences.
    int bc_start_idx = toml::find<int>(config, "first_index");
    int bc_end_idx = toml::find<int>(config, "last_index");

    if (bc_start_idx > bc_end_idx) {
        throw std::runtime_error("first_index must be <= last_index in the arrangement file.");
    }

    auto fill_bc_sequences = [bc_start_idx, bc_end_idx](const std::string& pattern,
                                                        std::vector<std::string>& bc_names) {
        auto modulo_pos = pattern.find_first_of("%");
        auto seq_name_prefix = pattern.substr(0, modulo_pos);
        auto format_str = pattern.substr(modulo_pos, pattern.length() - modulo_pos);

        for (int i = bc_start_idx; i <= bc_end_idx; i++) {
            char num[256];
            snprintf(num, 256, format_str.c_str(), i);
            bc_names.push_back(seq_name_prefix + std::string(num));
        }
    };

    // Fetch barcode 1 context (flanks + sequences).
    std::string barcode1_pattern = toml::find<std::string>(config, "barcode1_pattern");
    new_kit.top_front_flank = toml::find<std::string>(config, "mask1_front");
    new_kit.top_rear_flank = toml::find<std::string>(config, "mask1_rear");
    fill_bc_sequences(barcode1_pattern, new_kit.barcodes);

    // If any of the 2nd barcode settings are set, ensure ALL second barcode
    // settings are set.
    if (config.contains("mask2_front") || config.contains("mask2_rear") ||
        config.contains("barcode2_pattern")) {
        if (!(config.contains("mask2_front") && config.contains("mask2_rear") &&
              config.contains("barcode2_pattern"))) {
            throw std::runtime_error(
                    "For double ended barcodes, mask2_front mask2_rear and barcode2_pattern must "
                    "all be set.");
        }
        // Fetch barcode 2 context (flanks + sequences).
        new_kit.bottom_front_flank = toml::find<std::string>(config, "mask2_front");
        new_kit.bottom_rear_flank = toml::find<std::string>(config, "mask2_rear");
        std::string barcode2_pattern = toml::find<std::string>(config, "barcode2_pattern");

        fill_bc_sequences(barcode2_pattern, new_kit.barcodes2);

        new_kit.double_ends = true;
        new_kit.ends_different = (new_kit.bottom_front_flank != new_kit.top_front_flank) ||
                                 (new_kit.bottom_rear_flank != new_kit.top_rear_flank) ||
                                 (barcode1_pattern != barcode2_pattern);
    }

    return {kit_name, new_kit};
}

BarcodeKitScoringParams parse_scoring_params(const std::string& arrangement_file) {
    const toml::value config_toml = toml::parse(arrangement_file);

    if (!config_toml.contains("arrangement")) {
        throw std::runtime_error(
                "Custom barcode arrangement file must have [arrangement] section.");
    }

    const auto& config = toml::find(config_toml, "arrangement");

    BarcodeKitScoringParams params{};
    if (config.contains("min_soft_barcode_threshold")) {
        params.min_soft_barcode_threshold = toml::find<float>(config, "min_soft_barcode_threshold");
    }
    if (config.contains("min_hard_barcode_threshold")) {
        params.min_hard_barcode_threshold = toml::find<float>(config, "min_hard_barcode_threshold");
    }
    if (config.contains("min_soft_flank_threshold")) {
        params.min_soft_flank_threshold = toml::find<float>(config, "min_soft_flank_threshold");
    }
    if (config.contains("min_hard_flank_threshold")) {
        params.min_hard_flank_threshold = toml::find<float>(config, "min_hard_flank_threshold");
    }
    if (config.contains("min_barcode_score_dist")) {
        params.min_barcode_score_dist = toml::find<float>(config, "min_barcode_score_dist");
    }

    return params;
}

std::unordered_map<std::string, std::string> parse_custom_barcode_sequences(
        const std::string& sequences_file) {
    auto file = hts_open(sequences_file.c_str(), "r");
    BamPtr record;
    record.reset(bam_init1());

    std::unordered_map<std::string, std::string> sequences;

    while (sam_read1(file, nullptr, record.get()) >= 0) {
        std::string qname = bam_get_qname(record.get());
        std::string seq = utils::extract_sequence(record.get());
        sequences[qname] = seq;
    }

    hts_close(file);

    return sequences;
}

}  // namespace dorado::demux
