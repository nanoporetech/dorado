#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::barcode_kits {

struct KitInfo {
    std::string name;
    bool double_ends;
    bool ends_different;
    std::string top_front_flank;
    std::string top_rear_flank;
    std::string bottom_front_flank;
    std::string bottom_rear_flank;
    std::vector<std::string> barcodes;
    std::vector<std::string> barcodes2;
};

const std::unordered_map<std::string, KitInfo>& get_kit_infos();
const std::unordered_map<std::string, std::string>& get_barcodes();
const std::unordered_set<std::string>& get_barcode_identifiers();
std::string barcode_kits_list_str();

std::string normalize_barcode_name(const std::string& barcode_name);
std::string generate_standard_barcode_name(const std::string& kit_name,
                                           const std::string& barcode_name);
}  // namespace dorado::barcode_kits
