#pragma once

#include "utils/barcode_kits.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace dorado::demux {

std::optional<std::pair<std::string, barcode_kits::KitInfo>> parse_custom_arrangement(
        const std::string& arrangement_file);

barcode_kits::BarcodeKitScoringParams parse_scoring_params(
        const std::string& arrangement_file,
        const dorado::barcode_kits::BarcodeKitScoringParams& default_params);

bool check_normalized_id_pattern(const std::string& pattern);

}  // namespace dorado::demux
