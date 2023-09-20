#pragma once
#include "read_pipeline/ReadPipeline.h"

namespace dorado::utils {
ReadPtr shallow_copy_read(const SimplexRead& read);
}  // namespace dorado::utils
