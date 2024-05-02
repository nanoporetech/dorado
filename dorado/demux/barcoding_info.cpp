#include "barcoding_info.h"

#include <spdlog/spdlog.h>

namespace dorado::demux {

std::shared_ptr<const BarcodingInfo> create_barcoding_info(
        const std::vector<std::string>& kit_names,
        bool barcode_both_ends,
        bool trim_barcode,
        BarcodingInfo::FilterSet allowed_barcodes,
        const std::optional<std::string>& custom_kit,
        const std::optional<std::string>& custom_seqs) {
    if (kit_names.empty() && !custom_kit) {
        return {};
    }

    // Use either the kit name, or the custom kit path as the "kit name" specifier since
    // the custom kit's name is not determined till the kit is parsed.
    std::string kit_name = "";
    if (!kit_names.empty()) {
        kit_name = kit_names[0];
    }
    spdlog::debug("Creating barcoding info for kit: {}", kit_name);
    auto result =
            BarcodingInfo{kit_name,   barcode_both_ends, trim_barcode, std::move(allowed_barcodes),
                          custom_kit, custom_seqs};
    return std::make_shared<BarcodingInfo>(std::move(result));
}

}  // namespace dorado::demux