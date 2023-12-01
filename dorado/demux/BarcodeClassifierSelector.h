#pragma once
#include "utils/types.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dorado::demux {

class BarcodeClassifier;

class BarcodeClassifierSelector final {
    std::mutex m_mutex{};
    std::unordered_map<std::string, std::shared_ptr<const BarcodeClassifier>> m_barcoder_lut{};

public:
    std::shared_ptr<const BarcodeClassifier> get_barcoder(const BarcodingInfo& barcode_kit_info);
};

}  // namespace dorado::demux
