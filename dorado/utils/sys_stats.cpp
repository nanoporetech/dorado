#include "sys_stats.h"
#ifdef __linux__
#include <sys/resource.h>
#endif

namespace dorado {
namespace stats {

std::tuple<std::string, NamedStats> sys_stats_report() {
    stats::NamedStats named_stats;
#ifdef __linux__
    rusage usage;
    const int ret = getrusage(RUSAGE_SELF, &usage);
    if (ret == 0) {
        named_stats["maxrss_mb"] = usage.ru_maxrss / 1024;  // We are given this in KB.
    }
#endif
    return {"sys", named_stats};
}

}  // namespace stats
}  // namespace dorado
