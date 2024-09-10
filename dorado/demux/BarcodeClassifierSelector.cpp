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
    auto& barcoder = m_barcoder_lut[kit_id];
    if (!barcoder) {
        KitInfoProvider kit_info_provider(
                barcode_kit_info.kit_name.empty()
                        ? std::vector<std::string>{}
                        : std::vector<std::string>{barcode_kit_info.kit_name},
                barcode_kit_info.custom_kit, barcode_kit_info.custom_seqs);
        barcoder = std::make_shared<const BarcodeClassifier>(std::move(kit_info_provider));
    }
    return barcoder;
}

}  // namespace dorado::demux
