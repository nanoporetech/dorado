#pragma once
#include "read_pipeline/ReadPipeline.h"

namespace dorado::utils {
std::shared_ptr<Read> shallow_copy_read(const Read& read);
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves);
}  // namespace dorado::utils
