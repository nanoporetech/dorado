#pragma once
#include <cstddef>

namespace dorado::utils {

inline constexpr size_t BYTES_PER_GB{1024 * 1024 * 1024};

size_t available_host_memory_GB();
size_t total_host_memory_GB();

}  // namespace dorado::utils
