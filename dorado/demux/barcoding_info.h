#pragma once

#include "utils/types.h"

#include <string>

namespace dorado::demux {

struct BarcodingInfo {
    std::string kit_name;
    bool barcode_both_ends{false};
    bool trim{false};
    BarcodeFilterSet allowed_barcodes;
};

}  // namespace dorado::demux