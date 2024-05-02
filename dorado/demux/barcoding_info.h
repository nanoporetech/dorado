#pragma once

#include <optional>
#include <string>
#include <unordered_set>

namespace dorado::demux {

struct BarcodingInfo {
    using FilterSet = std::optional<std::unordered_set<std::string>>;
    std::string kit_name{};
    bool barcode_both_ends{false};
    bool trim{false};
    FilterSet allowed_barcodes;
    std::optional<std::string> custom_kit = std::nullopt;
    std::optional<std::string> custom_seqs = std::nullopt;
};

std::shared_ptr<const BarcodingInfo> create_barcoding_info(
        const std::vector<std::string> &kit_names,
        bool barcode_both_ends,
        bool trim_barcode,
        BarcodingInfo::FilterSet allowed_barcodes,
        const std::optional<std::string> &custom_kit,
        const std::optional<std::string> &custom_seqs);

}  // namespace dorado::demux