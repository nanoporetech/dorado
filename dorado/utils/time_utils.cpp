#include "utils/time_utils.h"

#include <chrono>
#include <sstream>

// Some stdlibs don't support parse()/from_stream() yet, or they can't format a sys_time<> correctly.
#if defined(__cpp_lib_format) && \
        ((defined(__GNUC__) && __GNUC__ >= 14) || (defined(__clang__) && defined(__linux__)))
#define FULL_CHRONO_SUPPORT 1
#else
#define FULL_CHRONO_SUPPORT 0
#endif

#if FULL_CHRONO_SUPPORT
#include <format>
namespace date = std::chrono;
#else
#include <date/date.h>
#include <date/tz.h>
#endif

namespace {

auto get_us_timepoint_from_ms(int64_t ms) {
    using namespace std::chrono;
    auto tp = system_clock::from_time_t(ms / 1000);
    tp += milliseconds(ms % 1000);
    auto delta = duration_cast<microseconds>(tp.time_since_epoch());
    return date::sys_time<microseconds>(delta);
}

auto get_timepoint_from_s(int64_t s) {
    using namespace std::chrono;
    auto tp = system_clock::from_time_t(s);
    auto delta = duration_cast<seconds>(tp.time_since_epoch());
    return date::sys_time<seconds>(delta);
}

}  // namespace

namespace dorado::utils {

std::string get_minknow_timestamp_from_unix_time_ms(int64_t ms) {
    auto tp = get_us_timepoint_from_ms(ms);
#if FULL_CHRONO_SUPPORT
    return std::format("{0:%Y%m%d}_{0:%H%M}", tp);
#else
    return date::format("%Y%m%d_%H%M", tp);
#endif
}

std::string get_string_timestamp_from_unix_time_ms(int64_t ms) {
    auto tp = get_us_timepoint_from_ms(ms);
#if FULL_CHRONO_SUPPORT
    return std::format("{0:%F}T{0:%T%Ez}", tp);
#else
    return date::format("%FT%T%Ez", tp);
#endif
}

std::string get_string_timestamp_from_unix_time_sZ(int64_t s) {
    auto tp = get_timepoint_from_s(s);
#if FULL_CHRONO_SUPPORT
    return std::format("{0:%F}T{0:%T}Z", tp);
#else
    return date::format("%FT%TZ", tp);
#endif
}

// Expects the time to be encoded like "2017-09-12T09:50:12.456+00:00" or "2017-09-12T09:50:12Z".
// Time stamp can be specified up to microseconds
int64_t get_unix_time_ms_from_string_timestamp(const std::string& time_stamp) {
    std::istringstream ss(time_stamp);
    date::sys_time<std::chrono::microseconds> time_us;
    ss >> date::parse("%FT%T%Ez", time_us);
    // If parsing with timezone offset failed, try parsing with 'Z' format
    if (ss.fail()) {
        ss.clear();
        ss.str(time_stamp);
        ss >> date::parse("%FT%TZ", time_us);
    }

    auto epoch = time_us.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    return value.count();
}

}  // namespace dorado::utils
