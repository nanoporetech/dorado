#include "memory_utils.h"

#if defined(WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include <array>

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

    auto avail_mem_GB = static_cast<size_t>(info.freeram) * info.mem_unit / BYTES_PER_GB;
    return avail_mem_GB;

#elif defined(__APPLE__)
    size_t unused_mem = 0;
    vm_size_t page_size;
    vm_statistics_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    mach_port_t mach_port = mach_host_self();
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS == host_statistics(mach_port, HOST_VM_INFO, (host_info_t)&vm_stats, &count)) {
        unused_mem = static_cast<size_t>(vm_stats.free_count) * page_size / BYTES_PER_GB;
    }
    return unused_mem;

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
