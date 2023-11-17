#include "BarcodeClassifierSelector.h"

#include "BarcodeClassifier.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <cassert>

namespace dorado::demux {

std::shared_ptr<const BarcodeClassifier> BarcodeClassifierSelector::get_barcoder(
        const BarcodingInfo& barcode_kit_info) {
    assert(!barcode_kit_info.kit_name.empty());
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_barcoder_lut.count(barcode_kit_info.kit_name)) {
        m_barcoder_lut.emplace(barcode_kit_info.kit_name,
                               std::make_shared<const BarcodeClassifier>(
                                       std::vector<std::string>{barcode_kit_info.kit_name},
                                       barcode_kit_info.custom_kit, barcode_kit_info.custom_seqs));
    }
    return m_barcoder_lut.at(barcode_kit_info.kit_name);
}

}  // namespace dorado::demux
