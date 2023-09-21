#pragma once

#include <optional>
#include <string>

namespace dorado::utils {

/**
 * Get the installed NVIDIA driver version.
 * @return std::nullopt if no driver, otherwise the version string.
 */
std::optional<std::string> get_nvidia_driver_version();

/// Implementation details, exposed for testing.
namespace detail {
std::optional<std::string> parse_nvidia_version_line(std::string_view line);
}  // namespace detail

}  // namespace dorado::utils
