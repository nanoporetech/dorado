#pragma once

#include "utils/SampleSheet.h"
#include "utils/types.h"

#include <string>

namespace dorado::demux {

struct BarcodingInfo {
    std::string kit_name;
    bool barcode_both_ends{false};
    bool trim{false};
    int max_barcode_errors{-1};  // -1 = use normal flank-based scoring; >= 0 = fuzzy edit-distance matching
    BarcodeFilterSet allowed_barcodes;
    std::shared_ptr<const utils::SampleSheet> sample_sheet;
};

}  // namespace dorado::demux
