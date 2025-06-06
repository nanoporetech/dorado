#pragma once

#include "utils/barcode_kits.h"

#include <string>
#include <unordered_map>

namespace dorado::demux {

std::pair<std::string, barcode_kits::KitInfo> parse_custom_arrangement(
        const std::string& arrangement_file);

barcode_kits::BarcodeKitScoringParams parse_scoring_params(
        const std::string& arrangement_file,
        const dorado::barcode_kits::BarcodeKitScoringParams& default_params);

bool check_normalized_id_pattern(const std::string& pattern);

std::pair<std::string, dorado::barcode_kits::KitInfo> get_custom_barcode_kit_info(
        const std::string& custom_kit_file);

}  // namespace dorado::demux
