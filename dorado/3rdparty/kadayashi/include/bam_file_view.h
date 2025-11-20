#pragma once

#include "hts_types.h"

namespace kadayashi {

struct BamFileView {
    htsFile* fp = nullptr;
    hts_idx_t* idx = nullptr;
    sam_hdr_t* hdr = nullptr;
};

}  // namespace kadayashi