#include "BarcodeClassifierSelector.h"

namespace dorado::demux {

BarcodeClassifier* BarcodeClassifierSelector::get_barcoder(const std::string& barcode_kit) {
    std::lock_guard<std::mutex> lock(m_mutex);
    assert(!barcode_kit.empty());
    if (!m_barcoder_lut.count(barcode_kit)) {
        std::vector<std::string> kits{barcode_kit};
        m_barcoder_lut.emplace(barcode_kit, BarcodeClassifier{kits});
    }
    return &m_barcoder_lut.at(barcode_kit);
}

}  // namespace dorado::demux