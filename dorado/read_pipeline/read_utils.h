#pragma once

#include "read_pipeline/messages.h"

namespace dorado::utils {
SimplexReadPtr shallow_copy_read(const SimplexRead& read);
}  // namespace dorado::utils
