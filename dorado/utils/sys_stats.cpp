#include "sys_stats.h"
#ifdef __linux__
#include <sys/resource.h>
#elif __APPLE__
#include <mach/mach_init.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#endif

namespace dorado {
namespace stats {

ReportedStats sys_stats_report() {
    stats::NamedStats named_stats;
#ifdef __linux__
    rusage usage;
    const int ret = getrusage(RUSAGE_SELF, &usage);
    if (ret == 0) {
        named_stats["maxrss_mb"] =
                static_cast<double>(usage.ru_maxrss) / 1024.0;  // We are given this in KB.
    }
#elif __APPLE__
    task_basic_info info;
    mach_msg_type_number_t info_count = TASK_BASIC_INFO_COUNT;
    // task_info's third parameter is of type task_info_t, an alias for integer_t.
    // If we specify TASK_BASIC_INFO, this is interpreted as a pointer to a
    // task_basic_info struct.
    kern_return_t ret = task_info(current_task(), TASK_BASIC_INFO,
                                  reinterpret_cast<task_info_t>(&info), &info_count);
    if (ret == KERN_SUCCESS) {
        // Memory sizes are given in bytes.
        named_stats["resident_size_mb"] =
                static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
        named_stats["virtual_size_mb"] = static_cast<double>(info.virtual_size) / (1024.0 * 1024.0);
    }
#endif

    return {"sys", named_stats};
}

}  // namespace stats
}  // namespace dorado
