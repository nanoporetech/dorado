#pragma once

#include "hts_utils/hts_types.h"

#include <string>
#include <string_view>

namespace dorado::utils {

struct ReadGroupData {
    bool found{false};  // True if any of the fields were filled.
    std::string id{};   // Read group ID.
    ReadGroup data{};   // Read group data.
};

ReadGroupData parse_rg_from_hts_tags(const std::string_view tag_str);

}  // namespace dorado::utils
