#pragma once

#include "utils/barcode_kits.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::demux {

class KitInfoProvider {
public:
    KitInfoProvider(const std::vector<std::string>& kit_names,
                    const std::optional<std::string>& custom_kit,
                    const std::optional<std::string>& custom_sequences);

    const barcode_kits::KitInfo& get_kit_info(const std::string& kit_name) const;
    const std::string& get_barcode_sequence(const std::string& barcode_name) const;
    barcode_kits::BarcodeKitScoringParams scoring_params() const { return m_scoring_params; };
    std::vector<std::string> kit_names() const { return m_kit_names; };

private:
    const std::unordered_map<std::string, dorado::barcode_kits::KitInfo> m_custom_kit;
    const std::unordered_map<std::string, std::string> m_custom_seqs;
    const barcode_kits::BarcodeKitScoringParams m_scoring_params;
    std::vector<std::string> m_kit_names;
};

}  // namespace dorado::demux
