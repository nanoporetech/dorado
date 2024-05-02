#include "BarcodeClassifierSelector.h"

#include "BarcodeClassifier.h"
#include "barcoding_info.h"

#include <spdlog/spdlog.h>

#include <cassert>

namespace dorado::demux {

std::shared_ptr<const BarcodeClassifier> BarcodeClassifierSelector::get_barcoder(
        const BarcodingInfo& barcode_kit_info) {
    if (barcode_kit_info.kit_name.empty() && !barcode_kit_info.custom_kit.has_value()) {
        throw std::runtime_error("Either kit name or custom kit file must be specified!");
    }
    const auto kit_id = barcode_kit_info.kit_name.empty() ? *barcode_kit_info.custom_kit
                                                          : barcode_kit_info.kit_name;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_barcoder_lut.count(kit_id)) {
        m_barcoder_lut.emplace(
                kit_id, std::make_shared<const BarcodeClassifier>(
                                barcode_kit_info.kit_name.empty()
                                        ? std::vector<std::string>{}
                                        : std::vector<std::string>{barcode_kit_info.kit_name},
                                barcode_kit_info.custom_kit, barcode_kit_info.custom_seqs));
    }
    return m_barcoder_lut.at(kit_id);
}

}  // namespace dorado::demux
