#pragma once

#include "Minimap2Options.h"

#include <string>

namespace dorado::alignment {

struct AlignmentInfo {
    alignment::Minimap2Options minimap_options;
    std::string reference_file;
};

}  // namespace dorado::alignment