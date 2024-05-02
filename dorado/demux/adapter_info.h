#pragma once

#include <optional>
#include <string>

namespace dorado::demux {

struct AdapterInfo {
    bool trim_adapters{true};
    bool trim_primers{true};
    std::optional<std::string> custom_seqs = std::nullopt;
};

}  // namespace dorado::demux