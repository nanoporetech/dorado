#pragma once
#include <spdlog/spdlog.h>

namespace dorado::utils {

// Initialises the default logger to point to stderr.
void InitLogging();

enum class VerboseLogLevel : int {
    none = 0,
    debug = 1,
    trace = 2,
};

void SetVerboseLogging(VerboseLogLevel level);

/// <summary>
/// Ensure a minimum logging level of info and increase verbosity according to the given VerboseLogLevel
/// </summary>
/// <remarks>
/// This is necessary for minKNOW to ensure progress stats are emmitted in case the
/// InitLogging fails the "is_safe_to_log" check and logging has been disabled.
/// MinKNOW does something along the line of redirecting stderr to stdout and then redirecting
/// stdout to a file
/// </remarks>
void EnsureInfoLoggingEnabled(VerboseLogLevel level);

/// Note that enabling this can impact short-read and adaptive-sampling performance.
#ifdef ENABLE_PER_READ_TRACE
template <typename... Args>
void trace_log(Args &&...args) {
    spdlog::trace(std::forward<Args>(args)...);
}
#else  // Per-read trace logging is disabled.
template <typename... Args>
void trace_log(Args &&...) {}
#endif

}  // namespace dorado::utils
