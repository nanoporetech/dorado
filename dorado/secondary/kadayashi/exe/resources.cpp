#include "resources.h"

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__) || defined(_POSIX_VERSION)
#include <sys/resource.h>
#endif

#include <chrono>
#include <cstddef>
#include <cstdlib>

namespace kadayashi {

bool KDY_VERBOSE = false;

double get_timestamp() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count()) /
           1000.0;
}

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__) || defined(_POSIX_VERSION)
double get_peakrss() {
    struct rusage s;
    getrusage(RUSAGE_SELF, &s);
    return (double)s.ru_maxrss / 1048576.0;  // GB
}
#else
double get_peakrss() { return 0.0; }
#endif

}  // namespace kadayashi
