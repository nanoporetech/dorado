#pragma once

#include <optional>
#include <string>

namespace dorado::utils {

/**
 * Get the installed NVIDIA driver version.
 * @return std::nullopt if no driver, otherwise the version string.
 */
std::optional<std::string> get_nvidia_driver_version();

}  // namespace dorado::utils
