#include "utils/memory_utils.h"

#include <spdlog/spdlog.h>

#if defined(WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include <array>
#include <fstream>
#include <sstream>

namespace dorado::utils {

double available_host_memory_GB() {
#if defined(WIN32)
    /**
     * The Windows API below already returns the total available (usable) memory for this application.
     */

    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<double>(status.ullAvailPhys) / BYTES_PER_GB;

#elif defined(__linux__)
    /**
     * On Linux, the Available memory consists of Free memory, Cached memory and Reclaimable memory.
     * In this case, “Free” memory is usually just the minor portion. There is the `sysinfo()` API (used
     * until now) which does not provide a way to get the info about the cached and the reclaimable memory.
     * The only way to get the actually available memory on Linux is to parse the `/proc/meminfo` file
     * and load the MemAvailable value.
     */

    std::ifstream ifs("/proc/meminfo");
    if (!ifs) {
        spdlog::warn("Cannot open /proc/meminfo to get memory statistics!");
        return 0.0;
    }

    uint64_t mem_available_kB = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string key, unit;
            iss >> key >> mem_available_kB >> unit;
            break;
        }
    }

    const double ret = mem_available_kB * 1024.0 / BYTES_PER_GB;

    return ret;

#elif defined(__APPLE__)
    /**
     * Mac has a slightly different way of handling memory than Linux. It also has several categories:
     *   - Free memory - only the currently free/unused memory. Can be pretty small.
     *   - Active memory - currently used by applications.
     *   - Inactive memory - not used by applications but left in cache for performance reasons if needed. This memory can be reclaimed by the system (though not instantly unlike the free memory).
     *   - Wired memory - system memory, not available to us.
     *   - Compressed memory - Also used by apps, compressed to avoid swapping as much as possible.
     *
     * To get the actual usable memory, we need to sum-up both the Free memory and the Inactive memory.
     */

    vm_size_t page_size;
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t mach_port = mach_host_self();
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS ==
                host_statistics64(mach_port, HOST_VM_INFO, (host_info_t)&vm_stats, &count)) {
        const uint64_t free_memory = static_cast<uint64_t>(vm_stats.free_count) * page_size;
        const uint64_t inactive_memory = static_cast<uint64_t>(vm_stats.inactive_count) * page_size;
        return static_cast<double>(free_memory + inactive_memory) / BYTES_PER_GB;
    }
    return 0.0;

#else
#error "Unsupported platform"
#endif
}

size_t total_host_memory_GB() {
#if defined(WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullTotalPhys) / BYTES_PER_GB;

#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) < 0) {
        return 0;
    }
    return static_cast<size_t>(info.totalram) * info.mem_unit / BYTES_PER_GB;

#elif defined(__APPLE__)
    std::array name{CTL_HW, HW_MEMSIZE};
    uint64_t total_size = 0;
    size_t length = sizeof(total_size);
    if (sysctl(name.data(), name.size(), &total_size, &length, nullptr, 0) != -1) {
        return total_size / BYTES_PER_GB;
    }
    return 0;

#else
#error "Unsupported platform"
#endif
}

}  // namespace dorado::utils
