#pragma once

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace dorado::utils::gpu_monitor {

/**
 * Get the installed NVIDIA driver version.
 * @return std::nullopt if no driver, otherwise the version string.
 */
std::optional<std::string> get_nvidia_driver_version();

struct DeviceStatusInfo {
    unsigned int device_index;

    std::optional<std::string> device_name;
    std::string device_name_error;

    // Bit mask representing causes of any current throttling of the device.
    // See NVML documentation for further information on nvmlClocksThrottleReasons
    std::optional<unsigned long long> current_throttling_reason;
    std::string current_throttling_reason_error;

    std::optional<unsigned int> current_temperature;
    std::string current_temperature_error;

    std::optional<unsigned int> gpu_shutdown_temperature;  // Temperature at which the GPU will
                                                           // shut down for HW protection
    std::string gpu_shutdown_temperature_error;

    std::optional<unsigned int> gpu_slowdown_temperature;  // Temperature at which the GPU will
                                                           // begin HW slowdown
    std::string gpu_slowdown_temperature_error;

    std::optional<unsigned int> gpu_max_operating_temperature;  // GPU Temperature at which the GPU
                                                                // can be throttled below base clock
    std::string gpu_max_operating_temperature_error;

    std::optional<unsigned int> current_power_usage;  // Current power usage in milliwatts
    std::string current_power_usage_error;

    std::optional<unsigned int>
            default_power_cap;  // Default max power limit in milliwatts before power
                                // management kicks in (this is the inital limit the
                                // device boots with).
    std::string default_power_cap_error;

    std::optional<unsigned int>
            percentage_utilization_gpu;  // Percent of time over the past sample period during which one or more kernels was executing on the GPU
    std::optional<unsigned int>
            percentage_utilization_memory;  // Percent of time over the past sample period during which global (device) memory was being read or written
    std::string
            percentage_utilization_error;  // Shared error reason retrieving utilization info (gpu and memory)

    std::optional<unsigned int>
            current_performance_state;  // 0 (max) .. 15 (min) performance. 32 represents unknown performance.
    std::string current_performance_state_error;
};

std::optional<DeviceStatusInfo> get_device_status_info(unsigned int device_index);

std::vector<std::optional<DeviceStatusInfo>> get_devices_status_info();

unsigned int get_device_count();

/// Implementation details, exposed for testing.
namespace detail {
std::optional<std::string> parse_nvidia_version_line(std::string_view line);
std::optional<std::string> parse_nvidia_tegra_line(const std::string& line);

// check whether the given device can queried via NVML
bool is_accessible_device(unsigned int device_index);
}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
