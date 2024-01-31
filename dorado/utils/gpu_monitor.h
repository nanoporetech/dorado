#pragma once

#include <optional>
#include <string>
#include <vector>

namespace dorado::utils::gpu_monitor {

/**
 * Get the installed NVIDIA driver version.
 * @return std::nullopt if no driver, otherwise the version string.
 */
std::optional<std::string> get_nvidia_driver_version();

//struct TemperatureInfo {
//    unsigned int device_index;
//    unsigned int current_temperature;
//
//    //nvmlDeviceGetCurrentClocksThrottleReasons
//    unsigned long long current_throttling_reason;
//
//    //nvmlDeviceGetPowerManagementMode
//    //bool is_power_management_system_active;
//};
//
//std::optional<std::vector<TemperatureInfo>> get_device_temperature_info(int device_id);

/// Implementation details, exposed for testing.
namespace detail {
std::optional<std::string> parse_nvidia_version_line(std::string_view line);

// Note returns total
std::optional<unsigned int> get_device_count();
}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
