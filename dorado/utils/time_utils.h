#pragma once
#include <date/date.h>
#include <date/tz.h>

#include <chrono>
#include <ctime>
#include <sstream>
#include <string>

namespace dorado::utils {

inline std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms) {
    auto tp = std::chrono::system_clock::from_time_t(time_stamp_ms / 1000);
    tp += std::chrono::milliseconds(time_stamp_ms % 1000);
    auto dp = date::floor<date::days>(tp);
    auto time = date::make_time(std::chrono::duration_cast<std::chrono::milliseconds>(tp - dp));
    auto ymd = date::year_month_day{dp};

    std::ostringstream date_time_ss;
    date_time_ss << ymd << "T" << time << "+00:00";
    return date_time_ss.str();
}

// Expects the time to be encoded like "2017-09-12T09:50:12.456+00:00" or "2017-09-12T09:50:12Z".
// Time stamp can be specified up to microseconds
inline time_t get_unix_time_from_string_timestamp(const std::string& time_stamp) {
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

inline std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms) {
    return get_string_timestamp_from_unix_time(get_unix_time_from_string_timestamp(time_stamp) +
                                               offset_ms);
}

inline std::string adjust_time(const std::string& time_stamp, uint32_t offset) {
    // Expects the time to be encoded like "2017-09-12T9:50:12Z".
    // Adds the offset (in seconds) to the timeStamp.

    date::sys_time<std::chrono::seconds> time_s;
    std::istringstream ss(time_stamp);
    ss >> date::parse("%FT%TZ", time_s);
    time_s += std::chrono::seconds(offset);

    auto dp = date::floor<date::days>(time_s);
    auto time = date::make_time(time_s - dp);
    auto ymd = date::year_month_day{dp};

    std::ostringstream date_time_ss;
    date_time_ss << ymd << "T" << time << "Z";
    return date_time_ss.str();
}

}  // namespace dorado::utils
