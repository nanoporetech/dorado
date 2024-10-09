#include "BarcodeClassifierSelector.h"

#include "BarcodeClassifier.h"
#include "barcoding_info.h"

#include <spdlog/spdlog.h>

#include <cassert>

namespace dorado::demux {

std::shared_ptr<const BarcodeClassifier> BarcodeClassifierSelector::get_barcoder(
        const BarcodingInfo& barcode_kit_info) {
    if (barcode_kit_info.kit_name.empty()) {
        throw std::runtime_error("Kit name must be specified!");
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    auto& barcoder = m_barcoder_lut[barcode_kit_info.kit_name];
    if (!barcoder) {
        barcoder = std::make_shared<const BarcodeClassifier>(barcode_kit_info.kit_name);
    }
    return barcoder;
}

}  // namespace dorado::demux
