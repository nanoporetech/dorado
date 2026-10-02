#pragma once
#include <spdlog/spdlog.h>

// Wrapper for spdlog logging that avoid evaluating log messages if
// they're not enabled.
// |lvl| is one of spdlog_level below.
#define DORADO_LOG(lvl, ...)                                          \
    do {                                                              \
        if (spdlog::should_log(::dorado::utils::spdlog_level::lvl)) { \
            spdlog::lvl(__VA_ARGS__);                                 \
        }                                                             \
    } while (false)

namespace dorado::utils {

// Map our log levels to spdlog's so we can reuse the same function names.
namespace spdlog_level {
constexpr auto trace = spdlog::level::level_enum::trace;
constexpr auto debug = spdlog::level::level_enum::debug;
constexpr auto info = spdlog::level::level_enum::info;
constexpr auto warn = spdlog::level::level_enum::warn;
constexpr auto error = spdlog::level::level_enum::err;
}  // namespace spdlog_level

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

}  // namespace dorado::utils
