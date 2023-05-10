#pragma once
#include "read_pipeline/ReadPipeline.h"

namespace dorado::utils {
std::shared_ptr<Read> copy_read(const Read& read);
}  // namespace dorado::utils
