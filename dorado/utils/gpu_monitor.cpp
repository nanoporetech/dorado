#include "gpu_monitor.h"

#if defined(_WIN32) || defined(__linux__)
#define HAS_NVML 1
#else
#define HAS_NVML 0
#endif

#if HAS_NVML
#include <nvml.h>
#if defined(_WIN32)
#include <Windows.h>
#else  // _WIN32
#include <dlfcn.h>
#endif  // _WIN32
#endif  // HAS_NVML

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace dorado::utils::gpu_monitor {

namespace {

#if HAS_NVML
#ifdef NVML_DEVICE_NAME_V2_BUFFER_SIZE
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_V2_BUFFER_SIZE
#elif defined(NVML_DEVICE_NAME_BUFFER_SIZE)
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_BUFFER_SIZE
#endif

// Prefixless versions of symbols we use
// X(name, optional)
#define FOR_EACH_NVML_SYMBOL(X)                     \
    X(DeviceGetCount, false)                        \
    X(DeviceGetCount_v2, true)                      \
    X(DeviceGetCurrentClocksThrottleReasons, false) \
    X(DeviceGetHandleByIndex, false)                \
    X(DeviceGetHandleByIndex_v2, true)              \
    X(DeviceGetName, false)                         \
    X(DeviceGetPerformanceState, false)             \
    X(DeviceGetPowerManagementDefaultLimit, false)  \
    X(DeviceGetPowerUsage, false)                   \
    X(DeviceGetTemperature, false)                  \
    X(DeviceGetTemperatureThreshold, false)         \
    X(DeviceGetUtilizationRates, false)             \
    X(Init, false)                                  \
    X(Init_v2, true)                                \
    X(Shutdown, false)                              \
    X(SystemGetDriverVersion, false)                \
    X(ErrorString, false)                           \
    // line intentionally blank

/**
 * Handle to the NVML API.
 * Also provides a scoped wrapper around NVML API initialisation.
 */
class NVMLAPI {
    // Includes devices which cannot be accessed via NVML so need to check return codes
    // on individual device specific NVML function calls.
    unsigned int m_device_count{};

    // Platform specific library handling.
#ifdef _WIN32
    HINSTANCE m_handle = nullptr;
    bool platform_open() {
        m_handle = LoadLibraryA("nvml.dll");
        if (m_handle != nullptr) {
            return true;
        }

        // Search in other places that the documentation and other resources mentions.
        const char *win64_dir_env = getenv("ProgramW6432");
        const std::string win64_dir = win64_dir_env ? win64_dir_env : "C:";
        const std::string paths[] = {
                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml\\lib\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml\\lib\\nvml.dll",
        };
        for (const auto &path : paths) {
            m_handle = LoadLibraryA(path.c_str());
            if (m_handle != nullptr) {
                return true;
            }
        }
        return false;
    }
    void platform_close() {
        if (m_handle != nullptr) {
            FreeLibrary(m_handle);
            m_handle = nullptr;
        }
    }
    template <typename T>
    bool platform_load_symbol(T *&func_ptr, const char *name, bool optional) {
        func_ptr = reinterpret_cast<T *>(GetProcAddress(m_handle, name));
        if (func_ptr == nullptr && !optional) {
            spdlog::warn("Failed to load NVML symbol {}: {}", name, GetLastError());
            return false;
        }
        return true;
    }

#else   // _WIN32
    void *m_handle = nullptr;
    bool platform_open() {
        // Prioritise loading the versioned lib.
        for (const char *path : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
            m_handle = dlopen(path, RTLD_NOW);
            if (m_handle != nullptr) {
                return true;
            }
        }
        return false;
    }
    void platform_close() {
        if (m_handle != nullptr) {
            dlclose(m_handle);
            m_handle = nullptr;
        }
    }
    template <typename T>
    bool platform_load_symbol(T *&func_ptr, const char *name, bool optional) {
        func_ptr = reinterpret_cast<T *>(dlsym(m_handle, name));
        if (func_ptr == nullptr && !optional) {
            spdlog::warn("Failed to load NVML symbol {}: {}", name, dlerror());
            return false;
        }
        return true;
    }
#endif  // _WIN32

    // Add members for each function pointer.
#define GENERATE_MEMBER(name, optional)         \
    using name##_ptr = decltype(&::nvml##name); \
    name##_ptr m_##name = nullptr;
    FOR_EACH_NVML_SYMBOL(GENERATE_MEMBER)
#undef GENERATE_MEMBER

    bool load_symbols() {
#define LOAD_SYMBOL(name, optional)                                \
    if (!platform_load_symbol(m_##name, "nvml" #name, optional)) { \
        return false;                                              \
    }
        FOR_EACH_NVML_SYMBOL(LOAD_SYMBOL)
#undef LOAD_SYMBOL
        return true;
    }

    void clear_symbols() {
#define CLEAR_SYMBOL(name, optional) m_##name = nullptr;
        FOR_EACH_NVML_SYMBOL(CLEAR_SYMBOL)
#undef CLEAR_SYMBOL
    }

    void init() {
        if (!platform_open() || !load_symbols()) {
            spdlog::warn("Failed to load NVML");
            clear_symbols();
            platform_close();
            return;
        }

        // Fall back to the original nvmlInit() if _v2 doesn't exist.
        auto *do_init = m_Init_v2 ? m_Init_v2 : m_Init;
        nvmlReturn_t result = do_init();
        if (result != NVML_SUCCESS) {
            spdlog::warn("Failed to initialize NVML: {}", m_ErrorString(result));
            clear_symbols();
            platform_close();
        }
    }

    void shutdown() {
        if (m_Shutdown != nullptr) {
            m_Shutdown();
        }
        platform_close();
    }

    NVMLAPI() {
        init();
        set_device_count();
    }

    ~NVMLAPI() { shutdown(); }

    NVMLAPI(const NVMLAPI &) = delete;
    NVMLAPI &operator=(const NVMLAPI &) = delete;

    void set_device_count() {
        if (!m_handle) {
            return;
        }
        auto *device_count_op = m_DeviceGetCount_v2 ? m_DeviceGetCount_v2 : m_DeviceGetCount;
        auto result = device_count_op(&m_device_count);
        if (result != NVML_SUCCESS) {
            m_device_count = 0;
            spdlog::warn("Call to DeviceGetCount failed: {}", ErrorString(result));
        }
    }

public:
    static NVMLAPI *get() {
        static NVMLAPI api;
        return api.m_handle != nullptr ? &api : nullptr;
    }

    std::optional<nvmlDevice_t> get_device_handle(unsigned int device_index) {
        nvmlDevice_t device;
        auto *get_handle = m_DeviceGetHandleByIndex_v2 ? m_DeviceGetHandleByIndex_v2
                                                       : m_DeviceGetHandleByIndex;
        nvmlReturn_t result = get_handle(device_index, &device);
        if (result != NVML_SUCCESS) {
            return std::nullopt;
        }
        return device;
    }

    nvmlReturn_t SystemGetDriverVersion(char *version, unsigned int length) {
        return m_SystemGetDriverVersion(version, length);
    }

    nvmlReturn_t DeviceGetTemperature(const nvmlDevice_t &device,
                                      nvmlTemperatureSensors_t sensorType,
                                      unsigned int *temp) {
        return m_DeviceGetTemperature(device, sensorType, temp);
    }

    nvmlReturn_t DeviceGetTemperatureThreshold(const nvmlDevice_t &device,
                                               nvmlTemperatureThresholds_t thresholdType,
                                               unsigned int *temp) {
        return m_DeviceGetTemperatureThreshold(device, thresholdType, temp);
    }

    nvmlReturn_t DeviceGetPerformanceState(const nvmlDevice_t &device, unsigned int *limit) {
        nvmlPstates_t state;
        auto result = m_DeviceGetPerformanceState(device, &state);
        *limit = static_cast<unsigned int>(state);
        return result;
    }

    nvmlReturn_t DeviceGetPowerManagementDefaultLimit(const nvmlDevice_t &device,
                                                      unsigned int *limit) {
        return m_DeviceGetPowerManagementDefaultLimit(device, limit);
    }

    nvmlReturn_t DeviceGetPowerUsage(const nvmlDevice_t &device, unsigned int *power) {
        return m_DeviceGetPowerUsage(device, power);
    }

    nvmlReturn_t DeviceGetUtilizationRates(const nvmlDevice_t &device,
                                           nvmlUtilization_t *utilization) {
        return m_DeviceGetUtilizationRates(device, utilization);
    }

    nvmlReturn_t DeviceGetCurrentClocksThrottleReasons(const nvmlDevice_t &device,
                                                       unsigned long long *clocksThrottleReasons) {
        return m_DeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);
    }

    nvmlReturn_t DeviceGetName(const nvmlDevice_t &device, char *name, unsigned int length) {
        return m_DeviceGetName(device, name, length);
    }

    unsigned int get_device_count() { return m_device_count; }

    const char *ErrorString(nvmlReturn_t result) { return m_ErrorString(result); }
};

std::optional<std::string> read_version_from_nvml() {
    // See if we have access to the API.
    auto *nvml_api = NVMLAPI::get();
    if (nvml_api == nullptr) {
        // NVMLAPI will have reported a warning if we get here.
        return std::nullopt;
    }

    // Grab the driver version
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE + 1]{};
    nvmlReturn_t result =
            nvml_api->SystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        return version;
    } else {
        spdlog::warn("Failed to query driver version: {}", nvml_api->ErrorString(result));
        return std::nullopt;
    }
}

void assign_threshold_temp(NVMLAPI *nvml,
                           const nvmlDevice_t &device,
                           nvmlTemperatureThresholds_t thresholdType,
                           std::optional<unsigned int> &temp,
                           std::string &error_reason) {
    unsigned int value{};
    auto result = nvml->DeviceGetTemperatureThreshold(device, thresholdType, &value);
    if (result == NVML_SUCCESS) {
        temp = value;
    } else {
        error_reason = nvml->ErrorString(result);
    }
}

void set_threshold_temperatures(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
                          info.gpu_shutdown_temperature, info.gpu_shutdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
                          info.gpu_slowdown_temperature, info.gpu_slowdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
                          info.gpu_max_operating_temperature,
                          info.gpu_max_operating_temperature_error);
}

void set_current_power_usage(NVMLAPI *nvml, nvmlDevice_t &device, DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerUsage(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_power_usage = value;
    } else {
        info.current_power_usage_error = nvml->ErrorString(result);
    }
}

void set_power_cap(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerManagementDefaultLimit(device, &value);
    if (result == NVML_SUCCESS) {
        info.default_power_cap = value;
    } else {
        info.default_power_cap_error = nvml->ErrorString(result);
    }
}

void set_utilization(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
    nvmlUtilization_t utilization{};
    auto result = nvml->DeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        info.percentage_utilization_error = nvml->ErrorString(result);
        return;
    }
    info.percentage_utilization_gpu = utilization.gpu;
    info.percentage_utilization_memory = utilization.memory;
}

void set_current_performace(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
    unsigned int value;
    auto result = nvml->DeviceGetPerformanceState(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_performace_state = value;
    } else {
        info.current_performace_state_error = nvml->ErrorString(result);
    }
}

void set_current_throttling_reason(NVMLAPI *nvml,
                                   const nvmlDevice_t &device,
                                   DeviceStatusInfo &info) {
    unsigned long long reason{};
    auto result = nvml->DeviceGetCurrentClocksThrottleReasons(device, &reason);
    if (result == NVML_SUCCESS) {
        info.current_throttling_reason = std::move(reason);
    } else {
        info.current_throttling_reason_error = nvml->ErrorString(result);
    }
}

void set_current_temperature(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value);
    if (result == NVML_SUCCESS) {
        info.current_temperature = value;
    } else {
        info.current_temperature_error = nvml->ErrorString(result);
    }
}

void set_device_name(NVMLAPI *nvml, const nvmlDevice_t &device, DeviceStatusInfo &info) {
#ifdef ONT_NVML_BUFFER_SIZE
    char device_name[ONT_NVML_BUFFER_SIZE];
    auto result = nvml->DeviceGetName(device, device_name, ONT_NVML_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        info.device_name = device_name;
    } else {
        info.device_name_error = nvml->ErrorString(result);
    }
#else
    info.device_name_error = "NVML buffer size undefined";
#endif
}

#endif  // HAS_NVML

#if defined(__linux__)
std::optional<std::string> read_version_from_proc() {
    std::ifstream version_file("/proc/driver/nvidia/version",
                               std::ios_base::in | std::ios_base::binary);
    if (!version_file.is_open()) {
        spdlog::warn("No NVIDIA version file found in /proc");
        return std::nullopt;
    }

    // Parse the file line by line.
    std::string line;
    while (std::getline(version_file, line)) {
        auto info = detail::parse_nvidia_version_line(line);
        if (info.has_value()) {
            // We only expect there to be 1 version line, so we can return it immediately.
            return info;
        }
    }

    spdlog::warn("No version line found in /proc version file");
    return std::nullopt;
}
#endif

}  // namespace

#if HAS_NVML
std::optional<DeviceStatusInfo> get_device_status_info(int device_index) {
    auto nvml = NVMLAPI::get();
    if (!nvml) {
        return std::nullopt;
    }
    auto device = nvml->get_device_handle(device_index);
    if (!device) {
        return std::nullopt;
    }
    DeviceStatusInfo info{};
    info.device_index = device_index;
    set_current_temperature(nvml, *device, info);
    set_threshold_temperatures(nvml, *device, info);
    set_current_power_usage(nvml, *device, info);
    set_power_cap(nvml, *device, info);
    set_utilization(nvml, *device, info);
    set_current_performace(nvml, *device, info);
    set_current_throttling_reason(nvml, *device, info);
    set_device_name(nvml, *device, info);
    return info;
}
#else
std::optional<DeviceStatusInfo> get_device_status_info(int) { return std::nullopt; }
#endif

std::vector<std::optional<DeviceStatusInfo>> get_devices_status_info() {
#if HAS_NVML
    std::vector<std::optional<DeviceStatusInfo>> result{};
    const auto max_devices = detail::get_device_count();
    for (unsigned int device_index{}; device_index < max_devices; ++device_index) {
        result.push_back(get_device_status_info(device_index));
        auto status_info = get_device_status_info(device_index);
    }
    return result;
#else
    return {};
#endif
}

std::optional<std::string> get_nvidia_driver_version() {
    std::optional<std::string> version;
#if HAS_NVML
    if (!version) {
        version = read_version_from_nvml();
    }
#endif  // HAS_NVML
#if defined(__linux__)
    if (!version) {
        version = read_version_from_proc();
    }
#endif  // __linux__
    return version;
}

std::string to_string(ThrottleReason reason) {
    switch (reason) {
    case ThrottleReason::none:
        return "none";
    case ThrottleReason::gpu_idle:
        return "gpu_idle";
    case ThrottleReason::applications_clocks_setting:
        return "applications_clocks_setting";
    case ThrottleReason::sw_power_cap:
        return "sw_power_cap";
    case ThrottleReason::hw_slowdown:
        return "hw_slowdown";
    case ThrottleReason::sync_boost:
        return "sync_boost";
    case ThrottleReason::sw_thermal_slowdown:
        return "sw_thermal_slowdown";
    case ThrottleReason::hw_thermal_slowdown:
        return "hw_thermal_slowdown";
    case ThrottleReason::hw_power_brake_slowdown:
        return "hw_power_brake_slowdown";
    case ThrottleReason::display_clock_setting:
        return "display_clock_setting";
    case ThrottleReason::all_reasons:
        return "all_reasons";
    }
    // unreachable code, but prevents compiler warning:
    // error: control reaches end of non-void function [-Werror=return-type]
    throw std::runtime_error("unrecognised Result value");
}

std::ostream &operator<<(std::ostream &os, ThrottleReason result) {
    os << to_string(result);
    return os;
}

namespace detail {

std::optional<std::string> parse_nvidia_version_line(std::string_view line) {
    // Format is undocumented, but appears to be 2/3 parts that are double-space separated.
    // NVRM version: <module type>  <version string>  [extra info]
    constexpr std::string_view separator{"  "};

    // Check line prefix.
    constexpr std::string_view prefix{"NVRM version: "};
    if (line.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }

    // Find the splits.
    auto module_begin = prefix.size();
    auto module_end = line.find(separator, module_begin);
    if (module_end == line.npos) {
        return std::nullopt;
    }
    auto version_begin = module_end + separator.size();
    auto version_end = line.find(separator, version_begin);
    if (version_end == line.npos) {
        version_end = line.size();
    }

    // We have all the info we need.
    return std::string(line.substr(version_begin, version_end - version_begin));
}

unsigned int get_device_count() {
#if HAS_NVML
    auto nvml = NVMLAPI::get();
    return nvml ? nvml->get_device_count() : 0;
#else
    return 0;
#endif  // HAS_NVML
}

#if HAS_NVML
std::optional<unsigned int> get_device_current_temperature(unsigned int device_index) {
    auto nvml = NVMLAPI::get();
    if (!nvml) {
        return std::nullopt;
    }
    auto device_handle = nvml->get_device_handle(device_index);
    if (!device_handle) {
        return std::nullopt;
    }
    unsigned int temp{};
    auto result = nvml->DeviceGetTemperature(*device_handle, NVML_TEMPERATURE_GPU, &temp);
    if (result != NVML_SUCCESS) {
        return std::nullopt;
    }
    return temp;
}
#else
std::optional<unsigned int> get_device_current_temperature(unsigned int) { return std::nullopt; }
#endif  // HAS_NVML

#if HAS_NVML
bool is_accessible_device(unsigned int device_index) {
    auto nvml = NVMLAPI::get();
    if (!nvml) {
        return false;
    }
    if (device_index >= nvml->get_device_count()) {
        return false;
    }
    auto device_handle = nvml->get_device_handle(device_index);
    if (!device_handle) {
        return false;
    }
    return true;
}
#else
bool is_accessible_device(unsigned int) { return false; }
#endif  // HAS_NVML

}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
