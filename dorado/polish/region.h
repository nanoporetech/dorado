#pragma once

#include <cstdint>

namespace dorado::polisher {

struct Region {
    int32_t seq_id = -1;
    int64_t start = 0;
    int64_t end = -1;
};

}  // namespace dorado::polisher
