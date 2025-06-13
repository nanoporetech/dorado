#pragma once

#include "Minimap2Options.h"

#include <string>

namespace dorado::alignment {

struct AlignmentInfo {
    alignment::Minimap2Options minimap_options;
    std::string reference_file;
    std::string bed_file;
    std::string alignment_header;
};

}  // namespace dorado::alignment