#pragma once
#include "BarcodeClassifier.h"

#include <map>
#include <mutex>
#include <string>

namespace dorado::demux {
class BarcodeClassifierSelector final {
    std::mutex m_mutex{};
    std::map<std::string, demux::BarcodeClassifier> m_barcoder_lut{};

public:
    BarcodeClassifier* get_barcoder(const std::string& barcode_kit);
};

}  // namespace dorado::demux
