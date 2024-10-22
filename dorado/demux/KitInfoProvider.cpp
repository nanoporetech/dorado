#include "KitInfoProvider.h"

#include "utils/barcode_kits.h"

#include <stdexcept>

namespace dorado::demux {

KitInfoProvider::KitInfoProvider(const std::string& kit_name) : m_kit_names({kit_name}) {}

const barcode_kits::KitInfo& KitInfoProvider::get_kit_info(const std::string& kit_name) const {
    const auto& barcode_kit_infos = barcode_kits::get_kit_infos();
    const auto prebuilt_kit_iter = barcode_kit_infos.find(kit_name);
    if (prebuilt_kit_iter != barcode_kit_infos.cend()) {
        return prebuilt_kit_iter->second;
    }
    throw std::runtime_error("Could not find " + kit_name + " in pre-built or custom kits");
}

const std::string& KitInfoProvider::get_barcode_sequence(const std::string& barcode_name) const {
    const auto& barcodes_map = barcode_kits::get_barcodes();
    const auto prebuilt_seqs_iter = barcodes_map.find(barcode_name);
    if (prebuilt_seqs_iter != barcodes_map.cend()) {
        return prebuilt_seqs_iter->second;
    }
    throw std::runtime_error("Could not find " + barcode_name +
                             " in pre-built or custom barcode sequences");
}

}  // namespace dorado::demux
