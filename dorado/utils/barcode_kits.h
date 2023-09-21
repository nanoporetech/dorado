#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::barcode_kits {

struct KitInfo {
    bool double_ends;
    bool ends_different;
    std::string top_front_flank;
    std::string top_rear_flank;
    std::string bottom_front_flank;
    std::string bottom_rear_flank;
    std::vector<std::string> barcodes;
};

const std::unordered_map<std::string, KitInfo>& get_kit_infos();
const std::unordered_map<std::string, std::string>& get_barcodes();
std::string barcode_kits_list_str();

}  // namespace dorado::barcode_kits
