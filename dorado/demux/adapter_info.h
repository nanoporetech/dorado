#pragma once

#include <optional>
#include <string>

namespace dorado::demux {

struct AdapterInfo {
    bool trim_adapters{true};
    bool trim_primers{true};
    bool rna_adapters{false};
    std::optional<std::string> kit_name = std::nullopt;
    std::optional<std::string> custom_seqs = std::nullopt;
};

}  // namespace dorado::demux