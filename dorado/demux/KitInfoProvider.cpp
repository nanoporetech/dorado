#include "KitInfoProvider.h"

#include "parse_custom_kit.h"
#include "parse_custom_sequences.h"

#include <stdexcept>

namespace dorado::demux {

namespace {
// Helper function to convert the parsed custom kit tuple
// into an unordered_map to simplify searching for kit info during
// barcoding.
std::unordered_map<std::string, dorado::barcode_kits::KitInfo> process_custom_kit(
        const std::optional<std::string>& custom_kit) {
    std::unordered_map<std::string, dorado::barcode_kits::KitInfo> kit_map;
    if (custom_kit) {
        auto custom_arrangement = dorado::demux::parse_custom_arrangement(*custom_kit);
        if (custom_arrangement) {
            const auto& [kit_name, kit_info] = *custom_arrangement;
            kit_map[kit_name] = kit_info;
        }
    }
    return kit_map;
}

dorado::barcode_kits::BarcodeKitScoringParams set_scoring_params(
        const std::vector<std::string>& kit_names,
        const std::optional<std::string>& custom_kit) {
    dorado::barcode_kits::BarcodeKitScoringParams params{};

    if (!kit_names.empty()) {
        // If it is one of the pre-defined kits, override the default scoring
        // params with whatever is set for that specific kit.
        const auto& kit_name = kit_names[0];
        const auto& kit_info = barcode_kits::get_kit_infos();
        auto prebuilt_kit_iter = kit_info.find(kit_name);
        if (prebuilt_kit_iter != kit_info.end()) {
            params = prebuilt_kit_iter->second.scoring_params;
        }
    }
    if (custom_kit) {
        // If a custom kit is passed, parse it for any scoring
        // params that need to override the default params.
        return parse_scoring_params(*custom_kit, params);
    } else {
        return params;
    }
}
}  // namespace

KitInfoProvider::KitInfoProvider(const std::vector<std::string>& kit_names,
                                 const std::optional<std::string>& custom_kit,
                                 const std::optional<std::string>& custom_sequences)
        : m_custom_kit(process_custom_kit(custom_kit)),
          m_custom_seqs(custom_sequences ? parse_custom_sequences(*custom_sequences)
                                         : std::unordered_map<std::string, std::string>{}),
          m_scoring_params(set_scoring_params(kit_names, custom_kit)) {
    if (!m_custom_kit.empty()) {
        for (auto& [kit_name, _] : m_custom_kit) {
            m_kit_names.push_back(kit_name);
        }
    } else if (kit_names.empty()) {
        throw std::runtime_error(
                "Either custom kit must include kit arrangement or a kit name needs to be passed "
                "in.");
    } else {
        m_kit_names = kit_names;
    }
}

const barcode_kits::KitInfo& KitInfoProvider::get_kit_info(const std::string& kit_name) const {
    auto custom_kit_iter = m_custom_kit.find(kit_name);
    if (custom_kit_iter != m_custom_kit.end()) {
        return custom_kit_iter->second;
    }
    const auto& barcode_kit_infos = barcode_kits::get_kit_infos();
    auto prebuilt_kit_iter = barcode_kit_infos.find(kit_name);
    if (prebuilt_kit_iter != barcode_kit_infos.end()) {
        return prebuilt_kit_iter->second;
    }
    throw std::runtime_error("Could not find " + kit_name + " in pre-built or custom kits");
}

const std::string& KitInfoProvider::get_barcode_sequence(const std::string& barcode_name) const {
    auto custom_seqs_iter = m_custom_seqs.find(barcode_name);
    if (custom_seqs_iter != m_custom_seqs.end()) {
        return custom_seqs_iter->second;
    }
    const auto& barcodes_map = barcode_kits::get_barcodes();
    auto prebuilt_seqs_iter = barcodes_map.find(barcode_name);
    if (prebuilt_seqs_iter != barcodes_map.end()) {
        return prebuilt_seqs_iter->second;
    }
    throw std::runtime_error("Could not find " + barcode_name +
                             " in pre-built or custom barcode sequences");
}

}  // namespace dorado::demux
