#include "demux/parse_custom_kit.h"

#include "demux/parse_custom_sequences.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const std::string ARRANGEMENT_TOML_KEY{"arrangement"};
const std::string SCORING_TOML_KEY{"scoring"};
}  // namespace

namespace dorado::demux {

bool check_normalized_id_pattern(const std::string& pattern) {
    auto modulo_pos = pattern.find_first_of('%');
    // Check for the presence of the % specifier.
    if (modulo_pos == std::string::npos) {
        return false;
    }
    auto i_pos = pattern.find_first_of('i', modulo_pos);
    // Check for the presence of 'i' since only integers are allowed
    // and also ensure that's at the end of the string.
    if (i_pos == std::string::npos) {
        return false;
    }
    if (i_pos != pattern.length() - 1) {
        return false;
    }
    // Validate that all characters between % and i are digits.
    if (std::any_of(pattern.begin() + modulo_pos + 1, pattern.begin() + i_pos,
                    [](unsigned char c) { return !std::isdigit(c); })) {
        return false;
    }
    return true;
}

std::pair<std::string, barcode_kits::KitInfo> parse_custom_arrangement(
        const std::string& arrangement_file) {
    const toml::value config_toml = toml::parse(arrangement_file);

    barcode_kits::KitInfo new_kit{};
    new_kit.double_ends = false;
    new_kit.ends_different = false;
    new_kit.rear_only_barcodes = false;
    new_kit.rna_barcodes = false;

    const auto& config = toml::find(config_toml, ARRANGEMENT_TOML_KEY);
    std::string kit_name = toml::find<std::string>(config, "name");

    new_kit.name = toml::find<std::string>(config, "kit");

    // Determine barcode sequences.
    int bc_start_idx = toml::find<int>(config, "first_index");
    int bc_end_idx = toml::find<int>(config, "last_index");

    if (bc_start_idx > bc_end_idx) {
        throw std::runtime_error("first_index must be <= last_index in the arrangement file.");
    }

    auto fill_bc_sequences = [](const std::string& pattern, std::vector<std::string>& bc_names,
                                int start_idx, int end_idx) {
        if (!check_normalized_id_pattern(pattern)) {
            throw std::runtime_error("Barcode pattern must be prefix%\\d+i, e.g. BC%02i");
        }

        auto modulo_pos = pattern.find_first_of('%');
        auto seq_name_prefix = pattern.substr(0, modulo_pos);
        auto format_str = pattern.substr(modulo_pos);

        for (int i = start_idx; i <= end_idx; i++) {
            char num[256];
            snprintf(num, 256, format_str.c_str(), i);
            bc_names.push_back(seq_name_prefix + std::string(num));
        }
    };

    // Fetch barcode 1 context (flanks + sequences).
    std::string barcode1_pattern = toml::find<std::string>(config, "barcode1_pattern");
    new_kit.top_front_flank = toml::find<std::string>(config, "mask1_front");
    new_kit.top_rear_flank = toml::find<std::string>(config, "mask1_rear");
    if (new_kit.top_front_flank.empty() && new_kit.top_rear_flank.empty()) {
        throw std::runtime_error(
                "At least one of mask1_front or mask1_rear needs to be specified.");
    }
    fill_bc_sequences(barcode1_pattern, new_kit.barcodes, bc_start_idx, bc_end_idx);

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
        if (new_kit.bottom_front_flank.empty() && new_kit.bottom_rear_flank.empty()) {
            throw std::runtime_error(
                    "At least one of mask2_front or mask2_rear needs to be specified.");
        }
        std::string barcode2_pattern = toml::find<std::string>(config, "barcode2_pattern");

        fill_bc_sequences(barcode2_pattern, new_kit.barcodes2, bc_start_idx, bc_end_idx);

        new_kit.double_ends = true;
        new_kit.ends_different = (new_kit.bottom_front_flank != new_kit.top_front_flank) ||
                                 (new_kit.bottom_rear_flank != new_kit.top_rear_flank) ||
                                 (barcode1_pattern != barcode2_pattern);
    }

    // If any of the essential inner barcode settings are set, ensure they are all set correctly.
    if (config.contains("first_index_inner") || config.contains("last_index_inner") ||
        config.contains("mask1_mid") || config.contains("barcode_inner1_pattern")) {
        if (!(config.contains("first_index_inner") && config.contains("last_index_inner") &&
              config.contains("mask1_mid") && config.contains("barcode_inner1_pattern"))) {
            throw std::runtime_error(
                    "For dual barcodes, first_index_inner, last_index_inner, mask1_mid and "
                    "barcode_inner1_pattern must all be set.");
        }

        // Fetch inner barcode context
        std::string barcode_inner1_pattern =
                toml::find<std::string>(config, "barcode_inner1_pattern");
        int bc_inner_start_idx = toml::find<int>(config, "first_index_inner");
        int bc_inner_end_idx = toml::find<int>(config, "last_index_inner");
        new_kit.top_mid_flank = toml::find<std::string>(config, "mask1_mid");
        if (config.contains("mask1_mid_inner")) {
            new_kit.top_mid_flank_inner = toml::find<std::string>(config, "mask1_mid_inner");
            new_kit.mid_flank_split = true;
        }
        fill_bc_sequences(barcode_inner1_pattern, new_kit.barcodes_inner1, bc_inner_start_idx,
                          bc_inner_end_idx);

        // If any of the 2nd inner barcode settings are set, ensure ALL second barcode inner
        // settings are set.
        if (config.contains("mask2_mid") || config.contains("barcode_inner2_pattern")) {
            if (!(config.contains("mask2_mid") && config.contains("barcode_inner2_pattern"))) {
                throw std::runtime_error(
                        "For double ended inner barcodes, mask2_mid and barcode_inner2_pattern "
                        "must "
                        "both be set.");
            }
            if (!new_kit.double_ends) {
                throw std::runtime_error(
                        "Inner barcodes cannot be specified as double ended if the outer barcodes "
                        "are not double ended.");
            }
            // Note this limitation may be removed in future, so the ends_different check below
            //  is still useful to ensure future kits are self-consistent.
            if (new_kit.ends_different) {
                throw std::runtime_error(
                        "For dual barcodes, double-ended kits where both ends are not the same are "
                        "currently unsupported.");
            }
            // Note this limitation may be removed in future, so the mask2_mid_inner check below
            //  is still useful to ensure future kits are self-consistent.
            if (new_kit.mid_flank_split) {
                throw std::runtime_error(
                        "Split mid flanks cannot currently be used with double ended barcode "
                        "arrangements.");
            }
            // Fetch inner barcode 2 context.
            new_kit.bottom_mid_flank = toml::find<std::string>(config, "mask2_mid");
            if (config.contains("mask2_mid_inner")) {
                new_kit.bottom_mid_flank_inner = toml::find<std::string>(config, "mask2_mid_inner");
                if (!new_kit.mid_flank_split) {
                    throw std::runtime_error(
                            "For dual barcodes, if mask2_mid_inner is set, mask1_mid_inner must "
                            "also be set.");
                }
            }
            std::string barcode_inner2_pattern =
                    toml::find<std::string>(config, "barcode_inner2_pattern");
            fill_bc_sequences(barcode_inner2_pattern, new_kit.barcodes_inner2, bc_inner_start_idx,
                              bc_inner_end_idx);

            // Note that it would be possible to set ends_different in this case, but it's not clear
            //  if mismatching inners and matching outers would ever be needed in practice.  For now,
            //  it's better to error on this, as it's likely to be a misconfigured .toml.
            if (!new_kit.ends_different &&
                ((new_kit.bottom_mid_flank != new_kit.top_mid_flank) ||
                 (barcode_inner1_pattern != barcode_inner2_pattern) ||
                 (new_kit.bottom_mid_flank_inner != new_kit.top_mid_flank_inner))) {
                throw std::runtime_error(
                        "For dual barcodes, if the outer barcodes match, the inner barcodes must "
                        "also match.");
            }
        }
    }

    if (config.contains("rear_only_barcodes")) {
        new_kit.rear_only_barcodes = toml::find<bool>(config, "rear_only_barcodes");
    }

    if (config.contains("rna_barcodes")) {
        new_kit.rna_barcodes = toml::find<bool>(config, "rna_barcodes");
    }

    return std::make_pair(kit_name, new_kit);
}

dorado::barcode_kits::BarcodeKitScoringParams parse_scoring_params(
        const std::string& arrangement_file,
        const dorado::barcode_kits::BarcodeKitScoringParams& base_params) {
    const toml::value config_toml = toml::parse(arrangement_file);

    auto params = base_params;
    if (!config_toml.contains("scoring")) {
        return params;
    }

    const auto& config = toml::find(config_toml, SCORING_TOML_KEY);
    if (config.contains("max_barcode_penalty")) {
        params.max_barcode_penalty = toml::find<int>(config, "max_barcode_penalty");
    }
    if (config.contains("barcode_end_proximity")) {
        params.barcode_end_proximity = toml::find<int>(config, "barcode_end_proximity");
    }
    if (config.contains("min_barcode_penalty_dist")) {
        params.min_barcode_penalty_dist = toml::find<int>(config, "min_barcode_penalty_dist");
    }
    if (config.contains("min_separation_only_dist")) {
        params.min_separation_only_dist = toml::find<int>(config, "min_separation_only_dist");
    }
    if (config.contains("flank_left_pad")) {
        params.flank_left_pad = toml::find<int>(config, "flank_left_pad");
    }
    if (config.contains("flank_right_pad")) {
        params.flank_right_pad = toml::find<int>(config, "flank_right_pad");
    }
    if (config.contains("flank_left_pad_inner")) {
        params.flank_left_pad_inner = toml::find<int>(config, "flank_left_pad_inner");
    }
    if (config.contains("flank_right_pad_inner")) {
        params.flank_right_pad_inner = toml::find<int>(config, "flank_right_pad_inner");
    }
    if (config.contains("front_barcode_window")) {
        params.front_barcode_window = toml::find<int>(config, "front_barcode_window");
    }
    if (config.contains("rear_barcode_window")) {
        params.rear_barcode_window = toml::find<int>(config, "rear_barcode_window");
    }
    if (config.contains("min_flank_score")) {
        params.min_flank_score = toml::find<float>(config, "min_flank_score");
    }
    if (config.contains("midstrand_flank_score")) {
        params.midstrand_flank_score = toml::find<float>(config, "midstrand_flank_score");
    }

    return params;
}

std::pair<std::string, dorado::barcode_kits::KitInfo> get_custom_barcode_kit_info(
        const std::string& custom_kit_file) {
    auto custom_kit_info = dorado::demux::parse_custom_arrangement(custom_kit_file);
    custom_kit_info.second.scoring_params =
            parse_scoring_params(custom_kit_file, dorado::barcode_kits::BarcodeKitScoringParams{});
    return custom_kit_info;
}

bool try_configure_custom_barcode_sequences(const std::optional<std::string>& custom_seqs) {
    if (!custom_seqs) {
        return true;
    }

    try {
        std::unordered_map<std::string, std::string> custom_barcodes;
        auto custom_sequences = parse_custom_sequences(*custom_seqs);
        for (const auto& entry : custom_sequences) {
            custom_barcodes.emplace(std::make_pair(entry.name, entry.sequence));
        }
        barcode_kits::add_custom_barcodes(custom_barcodes);
    } catch (const std::exception& e) {
        spdlog::error("Unable to parse custom sequences file: '{}' - '{}'", *custom_seqs, e.what());
        return false;
    } catch (...) {
        spdlog::error("Unable to parse custom sequences file '{}'", *custom_seqs);
        return false;
    }
    return true;
}

bool try_configure_custom_barcode_arrangement(const std::optional<std::string>& custom_kit) {
    if (!custom_kit) {
        return true;
    }

    try {
        auto [kit_name, kit_info] = get_custom_barcode_kit_info(*custom_kit);
        barcode_kits::add_custom_barcode_kit(kit_name, kit_info);
    } catch (const std::exception& e) {
        spdlog::error("Unable to load custom barcode arrangement file: '{}' - '{}'", *custom_kit,
                      e.what());
        return false;
    } catch (...) {
        spdlog::error("Unable to load custom barcode arrangement file: '{}'", *custom_kit);
        return false;
    }
    return true;
}

}  // namespace dorado::demux
