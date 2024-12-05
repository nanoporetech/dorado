#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::barcode_kits {

struct BarcodeKitScoringParams {
    int max_barcode_penalty = 9;
    int barcode_end_proximity = 75;
    int min_barcode_penalty_dist = 3;
    int min_separation_only_dist = 6;
    int flank_left_pad = 5;
    int flank_right_pad = 10;
    int front_barcode_window = 175;
    int rear_barcode_window = 175;
    float min_flank_score = 0.5f;
    float midstrand_flank_score = 0.95f;
};

struct KitInfo {
    std::string name;
    bool double_ends;
    bool ends_different;
    bool rear_only_barcodes;
    std::string top_front_flank;
    std::string top_rear_flank;
    std::string bottom_front_flank;
    std::string bottom_rear_flank;
    std::vector<std::string> barcodes;
    std::vector<std::string> barcodes2;
    BarcodeKitScoringParams scoring_params;
};

const std::unordered_map<std::string, KitInfo>& get_kit_infos();
const KitInfo* get_kit_info(const std::string& kit_name);
const std::unordered_map<std::string, std::string>& get_barcodes();
const std::unordered_set<std::string>& get_barcode_identifiers();

void add_custom_barcode_kit(const std::string& kit_name, const KitInfo& kit_info);
void add_custom_barcodes(const std::unordered_map<std::string, std::string>& barcodes);

void clear_custom_barcode_kits();
void clear_custom_barcodes();

bool is_valid_barcode_kit(const std::string& kit_name);

std::string barcode_kits_list_str();

std::string normalize_barcode_name(const std::string& barcode_name);
std::string generate_standard_barcode_name(const std::string& kit_name,
                                           const std::string& barcode_name);
}  // namespace dorado::barcode_kits
