#pragma once

#include <cstdint>
#include <ctime>
#include <string>

namespace dorado::utils {

// Get a formatted datetime from ms since unix epoch. This effectively converts pod5 start time
// to MinKnow formatted datetimes as "YYYYMMDD_hhmm"
std::string get_minknow_timestamp_from_unix_time_ms(int64_t time_stamp_ms);

// Get a formatted datetime from ms since unix epoch as YYYY-MM-DDThh:mm:ss.uuu+00:00
std::string get_string_timestamp_from_unix_time_ms(int64_t time_stamp_ms);

// Expects the time to be encoded like "2017-09-12T09:50:12.456+00:00" or "2017-09-12T09:50:12Z".
// Time stamp can be specified up to microseconds
int64_t get_unix_time_ms_from_string_timestamp(const std::string& time_stamp);

std::string adjust_time_ms(const std::string& time_stamp, int64_t offset_ms);

std::string adjust_time(const std::string& time_stamp, int64_t offset);

double time_difference_seconds(const std::string& timestamp1, const std::string& timestamp2);

}  // namespace dorado::utils
