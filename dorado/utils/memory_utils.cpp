#include "memory_utils.h"

#if defined(WIN32)
#include <sysinfoapi.h>
#else
#include <sys/sysinfo.h>
#endif

namespace {
constexpr size_t BYTES_PER_GB{1024 * 1024 * 1024};
}

namespace dorado::utils {

size_t available_host_memory_GB() {
#if defined(WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys) / BYTES_PER_GB;

#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) < 0) {
        return 0;
    }

    auto avail_mem_GB = (info.freeram / BYTES_PER_GB) * info.mem_unit;
    return avail_mem_GB;

#else
    // APPLE - TODO
    // sysctlbyname?
    return 0;
#endif
}

}  // namespace dorado::utils