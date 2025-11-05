#pragma once

#include "utils/SampleSheet.h"
#include "utils/types.h"

#include <string>

namespace dorado::demux {

struct BarcodingInfo {
    std::string kit_name;
    bool barcode_both_ends{false};
    bool trim{false};
    BarcodeFilterSet allowed_barcodes;
    std::shared_ptr<const utils::SampleSheet> sample_sheet;
};

}  // namespace dorado::demux
