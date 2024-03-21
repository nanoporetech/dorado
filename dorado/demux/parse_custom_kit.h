#pragma once

#include "utils/barcode_kits.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace dorado::demux {

std::optional<std::pair<std::string, barcode_kits::KitInfo>> parse_custom_arrangement(
        const std::string& arrangement_file);

std::unordered_map<std::string, std::string> parse_custom_sequences(
        const std::string& sequences_file);

dorado::barcode_kits::BarcodeKitScoringParams parse_scoring_params(
        const std::string& arrangement_file,
        const dorado::barcode_kits::BarcodeKitScoringParams& default_params);

bool check_normalized_id_pattern(const std::string& pattern);

}  // namespace dorado::demux
