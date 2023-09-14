#include "driver_query.h"

#if defined(_WIN32) || defined(__linux__)
#define HAS_NVML 1
#else
#define HAS_NVML 0
#endif

#if HAS_NVML
#include <nvml.h>
#endif

#include <spdlog/spdlog.h>

namespace dorado::utils {

namespace {

#if HAS_NVML

// Scoped wrapper around NVML API initialisation.
class NVMLAPI {
    bool m_inited = false;

    NVMLAPI(const NVMLAPI &) = delete;
    NVMLAPI &operator=(const NVMLAPI &) = delete;

public:
    NVMLAPI() {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            spdlog::warn("Failed to initialize NVML: {}", nvmlErrorString(result));
        }
        m_inited = result == NVML_SUCCESS;
    }

    ~NVMLAPI() {
        if (m_inited) {
            nvmlShutdown();
        }
    }

    bool is_inited() const { return m_inited; }
};

bool init_nvml() {
    static NVMLAPI api;
    return api.is_inited();
}

#endif  // HAS_NVML

}  // namespace

std::optional<std::string> get_nvidia_driver_version() {
#if HAS_NVML
    if (!init_nvml()) {
        return std::nullopt;
    }

    // Grab the driver version
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE + 1]{};
    nvmlReturn_t result =
            nvmlSystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        spdlog::warn("Failed to query driver version: {}", nvmlErrorString(result));
        return std::nullopt;
    }

    return version;
#else   // HAS_NVML
    return std::nullopt;
#endif  // HAS_NVML
}

}  // namespace dorado::utils
