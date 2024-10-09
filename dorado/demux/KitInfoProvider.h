#pragma once

#include <string>
#include <vector>

namespace dorado::barcode_kits {
struct KitInfo;
}

namespace dorado::demux {

class KitInfoProvider {
public:
    KitInfoProvider(const std::string& kit_name);

    const barcode_kits::KitInfo& get_kit_info(const std::string& kit_name) const;
    const std::string& get_barcode_sequence(const std::string& barcode_name) const;
    std::vector<std::string> kit_names() const { return m_kit_names; };

private:
    const std::vector<std::string> m_kit_names;
};

}  // namespace dorado::demux
