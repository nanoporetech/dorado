#pragma once

#include <cstdint>

namespace dorado::polisher {

struct Interval {
    int32_t start = 0;
    int32_t end = 0;

    int32_t length() const { return end - start; }
};

}  // namespace dorado::polisher
