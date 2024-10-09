#pragma once

#include <optional>
#include <string>
#include <unordered_set>

namespace dorado::demux {

struct BarcodingInfo {
    using FilterSet = std::optional<std::unordered_set<std::string>>;
    std::string kit_name;
    bool barcode_both_ends{false};
    bool trim{false};
    FilterSet allowed_barcodes;
};

}  // namespace dorado::demux