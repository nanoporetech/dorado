#pragma once

#include "utils/barcode_kits.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace dorado::barcode_kits {

struct BarcodeKitScoringParams {
    float min_soft_barcode_threshold = 0.7f;
    float min_hard_barcode_threshold = 0.6f;
    float min_soft_flank_threshold = 0.7f;
    float min_hard_flank_threshold = 0.6f;
    float min_barcode_score_dist = 0.25f;
};

std::optional<std::pair<std::string, barcode_kits::KitInfo>> parse_custom_arrangement(
        const std::string& arrangement_file);

BarcodeKitScoringParams parse_scoring_params(const std::string& arrangement_file);

bool check_normalized_id_pattern(const std::string& pattern);

}  // namespace dorado::barcode_kits
