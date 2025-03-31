// For some reason operator<<(Half) is missing from the final link when the helper lib is built.
// This is a copy of the file Half.cpp which is meant to provide that functionality.
// This file is also used to tell CMake which language the torch helper lib is.
#include <torch/version.h>
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 6  // Assuming SemVer

#include <c10/util/Half.h>

#include <iostream>

namespace c10 {

static_assert(std::is_standard_layout<Half>::value, "c10::Half must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Half& value) {
    out << (float)value;
    return out;
}
}  // namespace c10

#elif TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 6
// Fixed in this version
#else
#error "Unexpected Torch version. Check that this implementation still matches"
#endif
