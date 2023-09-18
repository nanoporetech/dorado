#include "driver_query.h"

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

namespace dorado::utils {

namespace {

#if HAS_NVML

// Prefixless versions of symbols we use
// X(name, optional)
#define FOR_EACH_NVML_SYMBOL(X)      \
    X(Init, false)                   \
    X(Init_v2, true)                 \
    X(Shutdown, false)               \
    X(SystemGetDriverVersion, false) \
    X(ErrorString, false)            \
    // line intentionally blank

/**
 * Handle to the NVML API.
 * Also provides a scoped wrapper around NVML API initialisation.
 */
class NVMLAPI {
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

    NVMLAPI() { init(); }

    ~NVMLAPI() { shutdown(); }

    NVMLAPI(const NVMLAPI &) = delete;
    NVMLAPI &operator=(const NVMLAPI &) = delete;

public:
    static NVMLAPI *get() {
        static NVMLAPI api;
        return api.m_handle != nullptr ? &api : nullptr;
    }

    nvmlReturn_t SystemGetDriverVersion(char *version, unsigned int length) {
        return m_SystemGetDriverVersion(version, length);
    }

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

}  // namespace detail

}  // namespace dorado::utils
