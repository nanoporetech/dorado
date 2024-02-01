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

struct DeviceStatusInfo {
    unsigned int device_index;

    std::optional<std::string> device_name;
    std::string device_name_error;

    // Bit mask representing throttle reasons. See NVML documentation for nvmlClocksThrottleReasons
    // 0x0000 : No throttling
    // 0x0001 : GPU Idle (ckock dropped to idle state)
    // 0x0002 : Application clocks
    // 0x0004 : Power scaling algorithm, due to power usage
    // 0x0008 : Hardware slowdown. (e.g. high temperature or power draw)
    // 0x0010 : Sync boost. GPU in a sync group and being held at a lower clock
    // 0x0020 : SW thermal slowdown. GPU or memory temperature above max
    // 0x0040 : HW thermal slowdown. Temperature too high
    // 0x0080 : HW power brake slowdown. External power brake triggered
    // 0x0100 : GPU clocks are limited by current setting of Display clocks
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
            current_performace_state;  // 0 (max) .. 15 (min) performance. 32 represents unknown performance.
    std::string current_performace_state_error;
};

std::optional<DeviceStatusInfo> get_device_status_info(int device_index);

// N.B there may be less entries than detail::get_device_count if not all devices can be fully queried via NVML
std::vector<DeviceStatusInfo> get_accessible_devices_status_info();

/// Implementation details, exposed for testing.
namespace detail {
std::optional<std::string> parse_nvidia_version_line(std::string_view line);

unsigned int get_device_count();

std::optional<unsigned int> get_device_current_temperature(unsigned int device_index);

// check whether the given device can queried via NVML
bool is_accessible_device(unsigned int device_index);
}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
