#pragma once

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace dorado::utils::gpu_monitor {

// Reproduces the #defined nvml values for nvmlClocksThrottleReasons, so we don't need to expose nvml.h
enum class ThrottleReason : unsigned long long {
    // Ignore! Tag enabling SFINAE approach to supporting bitwise operators with this scoped enum
    bit_flags = 0x0000000000000000LL,

    /** Bit mask representing no clocks throttling
     *
     * Clocks are as high as possible.
     */
    none = 0x0000000000000000LL,

    /** Nothing is running on the GPU and the clocks are dropping to Idle state
     * \note This limiter may be removed in a later release
     */
    gpu_idle = 0x0000000000000001LL,

    /** GPU clocks are limited by current setting of applications clocks
     *
     * @see nvmlDeviceSetApplicationsClocks
     * @see nvmlDeviceGetApplicationsClock
     */
    applications_clocks_setting = 0x0000000000000002LL,

    /** SW Power Scaling algorithm is reducing the clocks below requested clocks
     *
     * @see nvmlDeviceGetPowerUsage
     * @see nvmlDeviceSetPowerManagementLimit
     * @see nvmlDeviceGetPowerManagementLimit
     */
    sw_power_cap = 0x0000000000000004LL,

    /** HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
     *
     * This is an indicator of:
     *   - temperature being too high
     *   - External Power Brake Assertion is triggered (e.g. by the system power supply)
     *   - Power draw is too high and Fast Trigger protection is reducing the clocks
     *   - May be also reported during PState or clock change
     *      - This behavior may be removed in a later release.
     *
     * @see nvmlDeviceGetTemperature
     * @see nvmlDeviceGetTemperatureThreshold
     * @see nvmlDeviceGetPowerUsage
     */
    hw_slowdown = 0x0000000000000008LL,

    /** Sync Boost
     *
     * This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
     * order to maximize performance per watt. All GPUs in the sync boost group
     * will boost to the minimum possible clocks across the entire group. Look at
     * the throttle reasons for other GPUs in the system to see why those GPUs are
     * holding this one at lower clocks.
     *
     */
    sync_boost = 0x0000000000000010LL,

    /** SW Thermal Slowdown
     *
     * This is an indicator of one or more of the following:
     *  - Current GPU temperature above the GPU Max Operating Temperature
     *  - Current memory temperature above the Memory Max Operating Temperature
     *
     */
    sw_thermal_slowdown = 0x0000000000000020LL,

    /** HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
     *
     * This is an indicator of:
     *   - temperature being too high
     *
     * @see nvmlDeviceGetTemperature
     * @see nvmlDeviceGetTemperatureThreshold
     * @see nvmlDeviceGetPowerUsage
     */
    hw_thermal_slowdown = 0x0000000000000040LL,

    /** HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
     *
     * This is an indicator of:
     *   - External Power Brake Assertion being triggered (e.g. by the system power supply)
     *
     * @see nvmlDeviceGetTemperature
     * @see nvmlDeviceGetTemperatureThreshold
     * @see nvmlDeviceGetPowerUsage
     */
    hw_power_brake_slowdown = 0x0000000000000080LL,

    /** GPU clocks are limited by current setting of Display clocks
     */
    display_clock_setting = 0x0000000000000100LL,

    all_reasons = gpu_idle | applications_clocks_setting | sw_power_cap | hw_slowdown | sync_boost |
                  sw_thermal_slowdown | hw_thermal_slowdown | hw_power_brake_slowdown |
                  display_clock_setting
};

std::ostream& operator<<(std::ostream& os, ThrottleReason reason);
std::string to_string(ThrottleReason reason);

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
    // @see ThrottleReason for possible reasons are of which there may be multiple set
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

std::optional<DeviceStatusInfo> get_device_status_info(int device_index);

std::vector<std::optional<DeviceStatusInfo>> get_devices_status_info();

/// Implementation details, exposed for testing.
namespace detail {
std::optional<std::string> parse_nvidia_version_line(std::string_view line);

unsigned int get_device_count();

std::optional<unsigned int> get_device_current_temperature(unsigned int device_index);

// check whether the given device can queried via NVML
bool is_accessible_device(unsigned int device_index);
}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
