#include "BarcodeClassifierSelector.h"

#include "BarcodeClassifier.h"

#include <cassert>

namespace dorado::demux {

std::shared_ptr<const BarcodeClassifier> BarcodeClassifierSelector::get_barcoder(
        const std::string& barcode_kit) {
    assert(!barcode_kit.empty());
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_barcoder_lut.count(barcode_kit)) {
        m_barcoder_lut.emplace(barcode_kit, std::make_shared<const BarcodeClassifier>(
                                                    std::vector<std::string>{barcode_kit}));
    }
    return m_barcoder_lut.at(barcode_kit);
}

}  // namespace dorado::demux