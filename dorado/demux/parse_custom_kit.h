#pragma once

#include "utils/barcode_kits.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace dorado::demux {

struct BarcodeKitScoringParams {
    int max_barcode_cost = 12;
    int barcode_end_proximity = 75;
    int min_barcode_score_dist = 3;
    int min_separation_only_dist = 6;
    int flank_left_pad = 5;
    int flank_right_pad = 10;
};

std::optional<std::pair<std::string, barcode_kits::KitInfo>> parse_custom_arrangement(
        const std::string& arrangement_file);

std::unordered_map<std::string, std::string> parse_custom_sequences(
        const std::string& sequences_file);

BarcodeKitScoringParams parse_scoring_params(const std::string& arrangement_file);

bool check_normalized_id_pattern(const std::string& pattern);

}  // namespace dorado::demux
