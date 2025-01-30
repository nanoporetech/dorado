#include "gpu_monitor.h"

#if defined(_WIN32) || defined(__linux__)
#define HAS_NVML 1
#else
#define HAS_NVML 0
#endif

#if HAS_NVML
#include "utils/scoped_trace_log.h"
#include "utils/string_utils.h"

#include <nvml.h>
#if defined(_WIN32)
#include <Windows.h>
#else  // _WIN32
#include <dlfcn.h>
#endif  // _WIN32
#if DORADO_ORIN || DORADO_TX2
#include <torch/torch.h>
#endif  // DORADO_ORIN || DORADO_TX2
#endif  // HAS_NVML

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace dorado::utils::gpu_monitor {

namespace {

#if HAS_NVML
#ifdef NVML_DEVICE_NAME_V2_BUFFER_SIZE
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_V2_BUFFER_SIZE
#elif defined(NVML_DEVICE_NAME_BUFFER_SIZE)
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_BUFFER_SIZE
#endif
static_assert(ONT_NVML_BUFFER_SIZE, "nvml buffer size must be defined");

// Prefixless versions of symbols we use
// X(name, optional)
#define FOR_EACH_NVML_SYMBOL(X)                     \
    X(DeviceGetCount, false)                        \
    X(DeviceGetCount_v2, true)                      \
    X(DeviceGetCurrentClocksThrottleReasons, false) \
    X(DeviceGetHandleByIndex, false)                \
    X(DeviceGetHandleByIndex_v2, true)              \
    X(DeviceGetHandleByUUID, false)                 \
    X(DeviceGetIndex, false)                        \
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
class NvmlApi final {
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
            spdlog::info("Failed to load NVML");
            clear_symbols();
            platform_close();
            return;
        }

        // Fall back to the original nvmlInit() if _v2 doesn't exist.
        auto *do_init = m_Init_v2 ? m_Init_v2 : m_Init;

        // We retry initialisation for a certain amount of time, to allow the driver to load on system startup
        auto start = std::chrono::system_clock::now();
        auto wait_seconds = std::chrono::seconds(10);
        nvmlReturn_t result;
        do {
            result = do_init();
            if (result == NVML_SUCCESS) {
                break;
            }
            spdlog::warn("Failed to initialize NVML: {}, retrying in 1s...", m_ErrorString(result));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } while ((std::chrono::system_clock::now() - start) < wait_seconds);

        if (result != NVML_SUCCESS) {
            spdlog::warn("Failed to initialize NVML after {} seconds: {}", wait_seconds.count(),
                         m_ErrorString(result));
            clear_symbols();
            platform_close();
        }
    }

    void shutdown() {
        if (m_Shutdown != nullptr) {
            m_Shutdown();
        }
        clear_symbols();
        platform_close();
    }

    NvmlApi(const NvmlApi &) = delete;
    NvmlApi &operator=(const NvmlApi &) = delete;

    // This slight design flaw is in place of having NvmlApi as a singleton.
    // Instead it is held as a member variable of the DeviceInfoCache singleton.
    // This is preferable to having dependencies between singletons.
    friend class DeviceInfoCache;
    NvmlApi() { init(); }

    ~NvmlApi() { shutdown(); }

public:
    bool is_loaded() { return m_handle != nullptr; }

    std::optional<nvmlDevice_t> get_device_handle(unsigned int device_index) {
        nvmlDevice_t device;
        auto *get_handle = m_DeviceGetHandleByIndex_v2 ? m_DeviceGetHandleByIndex_v2
                                                       : m_DeviceGetHandleByIndex;
        ScopedTraceLog log{__func__};
        nvmlReturn_t result = get_handle(device_index, &device);
        if (result != NVML_SUCCESS) {
            return std::nullopt;
        }
        return device;
    }

    nvmlReturn_t SystemGetDriverVersion(char *version, unsigned int length) {
        ScopedTraceLog log{__func__};
        return m_SystemGetDriverVersion(version, length);
    }

    nvmlReturn_t DeviceGetTemperature(const nvmlDevice_t &device,
                                      nvmlTemperatureSensors_t sensorType,
                                      unsigned int *temp) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetTemperature(device, sensorType, temp);
    }

    nvmlReturn_t DeviceGetTemperatureThreshold(const nvmlDevice_t &device,
                                               nvmlTemperatureThresholds_t thresholdType,
                                               unsigned int *temp) {
        ScopedTraceLog log{__func__};
        log.write("nvmlTemperatureThresholds_t: " + std::to_string(thresholdType));
        return m_DeviceGetTemperatureThreshold(device, thresholdType, temp);
    }

    nvmlReturn_t DeviceGetPerformanceState(const nvmlDevice_t &device, unsigned int *limit) {
        ScopedTraceLog log{__func__};
        nvmlPstates_t state;
        auto result = m_DeviceGetPerformanceState(device, &state);
        *limit = static_cast<unsigned int>(state);
        return result;
    }

    nvmlReturn_t DeviceGetPowerManagementDefaultLimit(const nvmlDevice_t &device,
                                                      unsigned int *limit) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetPowerManagementDefaultLimit(device, limit);
    }

    nvmlReturn_t DeviceGetPowerUsage(const nvmlDevice_t &device, unsigned int *power) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetPowerUsage(device, power);
    }

    nvmlReturn_t DeviceGetUtilizationRates(const nvmlDevice_t &device,
                                           nvmlUtilization_t *utilization) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetUtilizationRates(device, utilization);
    }

    nvmlReturn_t DeviceGetCurrentClocksThrottleReasons(const nvmlDevice_t &device,
                                                       unsigned long long *clocksThrottleReasons) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);
    }

    nvmlReturn_t DeviceGetName(const nvmlDevice_t &device, char *name, unsigned int length) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetName(device, name, length);
    }

    nvmlReturn_t DeviceGetCount(unsigned int *count) {
        auto *device_count_op = m_DeviceGetCount_v2 ? m_DeviceGetCount_v2 : m_DeviceGetCount;
        ScopedTraceLog log{__func__};
        return device_count_op(count);
    }

    const char *ErrorString(nvmlReturn_t result) { return m_ErrorString(result); }
};

void assign_threshold_temp(NvmlApi *nvml,
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

void retrieve_and_assign_threshold_temperatures(NvmlApi *nvml,
                                                const nvmlDevice_t &device,
                                                DeviceStatusInfo &info) {
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
                          info.gpu_shutdown_temperature, info.gpu_shutdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
                          info.gpu_slowdown_temperature, info.gpu_slowdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
                          info.gpu_max_operating_temperature,
                          info.gpu_max_operating_temperature_error);
}

void retrieve_and_assign_current_power_usage(NvmlApi *nvml,
                                             nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerUsage(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_power_usage = value;
    } else {
        info.current_power_usage_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_power_cap(NvmlApi *nvml,
                                   const nvmlDevice_t &device,
                                   DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerManagementDefaultLimit(device, &value);
    if (result == NVML_SUCCESS) {
        info.default_power_cap = value;
    } else {
        info.default_power_cap_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_utilization(NvmlApi *nvml,
                                     const nvmlDevice_t &device,
                                     DeviceStatusInfo &info) {
    nvmlUtilization_t utilization{};
    auto result = nvml->DeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        info.percentage_utilization_error = nvml->ErrorString(result);
        return;
    }
    info.percentage_utilization_gpu = utilization.gpu;
    info.percentage_utilization_memory = utilization.memory;
}

void retrieve_and_assign_current_performance(NvmlApi *nvml,
                                             const nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value;
    auto result = nvml->DeviceGetPerformanceState(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_performance_state = value;
    } else {
        info.current_performance_state_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_current_throttling_reason(NvmlApi *nvml,
                                                   const nvmlDevice_t &device,
                                                   DeviceStatusInfo &info) {
    unsigned long long reason{};
    auto result = nvml->DeviceGetCurrentClocksThrottleReasons(device, &reason);
    if (result == NVML_SUCCESS) {
        info.current_throttling_reason = reason;
    } else {
        info.current_throttling_reason_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_current_temperature(NvmlApi *nvml,
                                             const nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value);
    if (result == NVML_SUCCESS) {
        info.current_temperature = value;
    } else {
        info.current_temperature_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_device_name(NvmlApi *nvml,
                                     const nvmlDevice_t &device,
                                     DeviceStatusInfo &info) {
    char device_name[ONT_NVML_BUFFER_SIZE];
    auto result = nvml->DeviceGetName(device, device_name, ONT_NVML_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        info.device_name = device_name;
    } else {
        info.device_name_error = nvml->ErrorString(result);
    }
}

class DeviceHandles final {
    NvmlApi &m_nvml;
    std::unordered_map<unsigned int, std::optional<nvmlDevice_t>> m_device_handles{};

public:
    DeviceHandles(NvmlApi &nvml) : m_nvml(nvml) { assert(m_nvml.is_loaded()); }

    std::optional<nvmlDevice_t> get_handle(unsigned int device_index) {
        auto device_handle_lookup = m_device_handles.find(device_index);
        if (device_handle_lookup != m_device_handles.end()) {
            return device_handle_lookup->second;
        }

        auto device = m_nvml.get_device_handle(device_index);
        m_device_handles[device_index] = device;
        return device;
    }
};

class DeviceInfoCache final {
    std::mutex m_mutex{};
    NvmlApi m_nvml{};
    std::unique_ptr<DeviceHandles> m_device_handles;
    std::unordered_map<nvmlDevice_t, std::optional<DeviceStatusInfo>> m_device_info{};
    // Includes devices which cannot be accessed via NVML so need to check return codes
    // on individual device specific NVML function calls.
    unsigned int m_device_count = 0;
    std::vector<unsigned int> m_visible_device_indices;

    DeviceInfoCache() {
        set_device_count();
        if (m_nvml.is_loaded()) {
            m_device_handles = std::make_unique<DeviceHandles>(m_nvml);
        }
    }

    void map_visible_devices(unsigned int device_count) {
        // NVML doesn't respect CUDA_VISIBLE_DEVICES envvar, so check this separately
        const char *cuda_visible_devices_env = std::getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_visible_devices_env != nullptr) {
            spdlog::debug("Found CUDA_VISIBLE_DEVICES={}", cuda_visible_devices_env);
            std::set<int> used_ids;
            auto device_ids = utils::split(cuda_visible_devices_env, ',');
            if (device_ids.size() > device_count) {
                spdlog::error(
                        "CUDA_VISIBLE_DEVICES={} specifies more device ids than the number of GPUs "
                        "present",
                        cuda_visible_devices_env);
                throw std::runtime_error("Invalid device ids");
            }

            if (!device_ids.empty() && (utils::starts_with(device_ids.front(), "GPU-") ||
                                        utils::starts_with(device_ids.front(), "MIG-"))) {
                for (const auto &id : device_ids) {
                    nvmlDevice_t device;
                    if (auto rc = m_nvml.m_DeviceGetHandleByUUID(id.c_str(), &device);
                        rc != NVML_SUCCESS) {
                        spdlog::warn(
                                "Unable to identify GPU device '{}' - skipping further device "
                                "enumeration",
                                id);
                        // stop parsing on error
                        break;
                    }

                    unsigned int index = 0;
                    if (auto rc = m_nvml.m_DeviceGetIndex(device, &index); rc != NVML_SUCCESS) {
                        spdlog::warn(
                                "Unable to retrieve index for GPU device '{}' - skipping further "
                                "device enumeration",
                                id);
                        // stop parsing on error
                        break;
                    };
                    used_ids.insert(index);
                    m_visible_device_indices.push_back(index);
                }
            } else {
                std::string_view last_device_id;
                try {
                    for (const auto &id : device_ids) {
                        last_device_id = id;
                        int index = std::stoi(id);
                        if (index < 0 || index >= static_cast<int>(device_count)) {
                            spdlog::warn(
                                    "Invalid index '{}' for GPU device - skipping further device "
                                    "enumeration",
                                    index);
                            // stop parsing on invalid id
                            break;
                        }

                        used_ids.insert(index);
                        m_visible_device_indices.push_back(index);
                    }
                } catch (const std::exception &) {
                    // stop parsing on error
                    spdlog::warn(
                            "Unable to parse id '{}' for GPU device - skipping further device "
                            "enumeration",
                            last_device_id);
                }
            }
            if (used_ids.size() != m_visible_device_indices.size()) {
                // passing the same id twice should return no devices
                spdlog::warn("Duplicate GPU ids detected - no GPUs identified");
                m_visible_device_indices.clear();
            }
        } else {
            m_visible_device_indices.resize(device_count);
            std::iota(std::begin(m_visible_device_indices), std::end(m_visible_device_indices), 0);
        }
    }

    void set_device_count() {
        unsigned int device_count = 0;
        if (m_nvml.is_loaded()) {
            auto result = m_nvml.DeviceGetCount(&device_count);
            if (result != NVML_SUCCESS) {
                device_count = 0;
                spdlog::warn("Call to DeviceGetCount failed: {}", m_nvml.ErrorString(result));
            }
        }
#if DORADO_ORIN || DORADO_TX2
        if (device_count == 0) {
            // TX2/Orin may not have NVML, in which case ask torch how many devices it thinks there are.
            device_count = torch::cuda::device_count();
            spdlog::info("Setting device count to {} as reported from torch", device_count);
        }
#endif
        map_visible_devices(device_count);
        unsigned int cuda_visible_devices_count =
                static_cast<unsigned int>(m_visible_device_indices.size());

        if (cuda_visible_devices_count > device_count) {
            spdlog::warn(
                    "CUDA_VISIBLE_DEVICES contains more device ids ({}) than devices found by NVML "
                    "({}).",
                    cuda_visible_devices_count, device_count);
        }
        m_device_count = std::min(cuda_visible_devices_count, device_count);
    }

    std::optional<DeviceStatusInfo> create_new_device_entry(unsigned int device_index,
                                                            nvmlDevice_t device) {
        auto &info = m_device_info.emplace(device, DeviceStatusInfo{}).first->second;
        info->device_index = device_index;
        retrieve_and_assign_threshold_temperatures(&m_nvml, device, *info);
        retrieve_and_assign_power_cap(&m_nvml, device, *info);
        retrieve_and_assign_device_name(&m_nvml, device, *info);
        return info;
    }

    std::pair<std::optional<DeviceStatusInfo>, nvmlDevice_t> get_cached_device_info(
            unsigned int device_index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const unsigned int mapped_device_index = m_visible_device_indices[device_index];
        auto device = m_device_handles->get_handle(mapped_device_index);
        if (!device) {
            return {std::nullopt, nullptr};
        }

        auto device_info_lookup = m_device_info.find(*device);
        if (device_info_lookup != m_device_info.end()) {
            return {device_info_lookup->second, *device};
        }

        return {create_new_device_entry(mapped_device_index, *device), *device};
    }

public:
    static DeviceInfoCache &instance() {
        static DeviceInfoCache cache;
        return cache;
    }

    std::optional<nvmlDevice_t> get_device_handle(unsigned int device_index) {
        if (!m_nvml.is_loaded()) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_device_handles->get_handle(m_visible_device_indices[device_index]);
    }

    NvmlApi &nvml() { return m_nvml; }

    unsigned int get_device_count() { return m_device_count; }

    std::optional<DeviceStatusInfo> get_device_info(unsigned int device_index) {
        if (!m_nvml.is_loaded()) {
            return std::nullopt;
        }
        auto [info, device] = get_cached_device_info(device_index);
        if (!info) {
            return std::nullopt;
        }

        // We have a copy of the cached DeviceStatusInfo so we can update without
        // locking. NVML itelf is thread safe.
        retrieve_and_assign_current_temperature(&m_nvml, device, *info);
        retrieve_and_assign_current_power_usage(&m_nvml, device, *info);
        retrieve_and_assign_utilization(&m_nvml, device, *info);
        retrieve_and_assign_current_performance(&m_nvml, device, *info);
        retrieve_and_assign_current_throttling_reason(&m_nvml, device, *info);

        return info;
    }
};

std::optional<std::string> read_version_from_nvml() {
    auto &nvml_api = DeviceInfoCache::instance().nvml();
    if (!nvml_api.is_loaded()) {
        return std::nullopt;
    }

    // Grab the driver version
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE + 1]{};
    nvmlReturn_t result =
            nvml_api.SystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        return version;
    } else {
        spdlog::warn("Failed to query driver version: {}", nvml_api.ErrorString(result));
        return std::nullopt;
    }
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

#if DORADO_TX2
std::optional<std::string> read_version_from_tegra_release() {
    std::ifstream version_file("/etc/nv_tegra_release", std::ios_base::in | std::ios_base::binary);
    if (!version_file.is_open()) {
        spdlog::warn("No nv_tegra_release file found in /etc");
        return std::nullopt;
    }

    // First line should contain the version.
    std::string line;
    if (!std::getline(version_file, line)) {
        spdlog::warn("Failed to read first line from nv_tegra_release file");
        return std::nullopt;
    }

    auto info = detail::parse_nvidia_tegra_line(line);
    if (!info.has_value()) {
        spdlog::warn("Failed to parse version line from nv_tegra_release file: '{}'", line);
    }
    return info;
}

bool running_in_docker() {
    // Look for docker paths in the init process.
    std::ifstream cgroup_file("/proc/1/cgroup", std::ios_base::in | std::ios_base::binary);
    if (cgroup_file.is_open()) {
        std::string line;
        while (std::getline(cgroup_file, line)) {
            if (line.find(":/docker/") != line.npos) {
                return true;
            }
        }
    }
    return false;
}
#endif  // DORADO_TX2

}  // namespace

std::optional<DeviceStatusInfo> get_device_status_info(unsigned int device_index) {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_info(device_index);
#else
    (void)device_index;
    return std::nullopt;
#endif
}

std::vector<std::optional<DeviceStatusInfo>> get_devices_status_info() {
#if HAS_NVML
    std::vector<std::optional<DeviceStatusInfo>> result{};
    const auto max_devices = get_device_count();
    for (unsigned int device_index{}; device_index < max_devices; ++device_index) {
        result.push_back(get_device_status_info(device_index));
    }
    return result;
#else
    return {};
#endif
}

std::optional<std::string> get_nvidia_driver_version() {
    static auto cached_version = [] {
        std::optional<std::string> version;
#if HAS_NVML
        version = read_version_from_nvml();
#endif  // HAS_NVML
#if defined(__linux__)
        if (!version) {
            version = read_version_from_proc();
        }
#endif  // __linux__
#if DORADO_TX2
        if (!version) {
            version = read_version_from_tegra_release();
        }
        if (!version && running_in_docker()) {
            // The docker images we run in aren't representative of running natively on a
            // device, so we fake a version number to allow the tests to pass. On a real
            // machine we'll have grabbed the version from the tegra release file.
            spdlog::warn("Can't query version when running inside a docker container on TX2");
            version = "0.0.1";
        }
#endif  // DORADO_TX2
        return version;
    }();
    return cached_version;
}

unsigned int get_device_count() {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_count();
#else
    return 0;
#endif  // HAS_NVML
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

std::optional<std::string> parse_nvidia_tegra_line(const std::string &line) {
    // Based off of the following:
    // https://forums.developer.nvidia.com/t/how-do-i-know-what-version-of-l4t-my-jetson-tk1-is-running/38893

    // Simple regex should do it.
    const std::regex search("^# R(\\d+) \\(release\\), REVISION: (\\d+)\\.(\\d+)");
    std::smatch match;
    if (!std::regex_search(line, match, search)) {
        return std::nullopt;
    }

    // Reconstruct the version.
    return match[1].str() + "." + match[2].str() + "." + match[3].str();
}

bool is_accessible_device([[maybe_unused]] unsigned int device_index) {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_handle(device_index).has_value();
#else
    return false;
#endif  // HAS_NVML
}

}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
