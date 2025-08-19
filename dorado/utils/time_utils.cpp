#include "utils/time_utils.h"

#if __cplusplus >= 202002L && 0  // Most stdlibs don't support parse()/from_stream() yet
namespace date = std::chrono;
#else
#include <date/date.h>
#include <date/tz.h>
#endif

#include <chrono>
#include <sstream>

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
    return date::format("%Y%m%d_%H%M", tp);
}

std::string get_string_timestamp_from_unix_time_ms(int64_t ms) {
    auto tp = get_us_timepoint_from_ms(ms);
    return date::format("%FT%T%Ez", tp);
}

std::string get_string_timestamp_from_unix_time_sZ(int64_t s) {
    auto tp = get_timepoint_from_s(s);
    return date::format("%FT%TZ", tp);
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

std::string adjust_time_ms(const std::string& time_stamp, int64_t offset_ms) {
    return get_string_timestamp_from_unix_time_ms(
            get_unix_time_ms_from_string_timestamp(time_stamp) + offset_ms);
}

std::string adjust_time(const std::string& time_stamp, int64_t offset) {
    // Expects the time to be encoded like "2017-09-12T9:50:12Z".
    // Adds the offset (in seconds) to the timeStamp.

    date::sys_time<std::chrono::seconds> time_s;
    std::istringstream ss(time_stamp);
    ss >> date::parse("%FT%TZ", time_s);
    time_s += std::chrono::seconds(offset);

    auto dp = date::floor<date::days>(time_s);
    auto time = date::hh_mm_ss(time_s - dp);
    auto ymd = date::year_month_day{dp};

    std::ostringstream date_time_ss;
    date_time_ss << ymd << "T" << time << "Z";
    return date_time_ss.str();
}

double time_difference_seconds(const std::string& timestamp1, const std::string& timestamp2) {
    try {
        std::istringstream ss1(timestamp1);
        std::istringstream ss2(timestamp2);
        date::sys_time<std::chrono::microseconds> time1, time2;
        ss1 >> date::parse("%FT%T%Ez", time1);
        ss2 >> date::parse("%FT%T%Ez", time2);
        // If parsing with timezone offset failed, try parsing with 'Z' format
        if (ss1.fail()) {
            ss1.clear();
            ss1.str(timestamp1);
            ss1 >> date::parse("%FT%TZ", time1);
        }
        if (ss2.fail()) {
            ss2.clear();
            ss2.str(timestamp2);
            ss2 >> date::parse("%FT%TZ", time2);
        }
        std::chrono::duration<double> diff = time1 - time2;
        return diff.count();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse timestamps: ") + e.what());
    }
}

}  // namespace dorado::utils
