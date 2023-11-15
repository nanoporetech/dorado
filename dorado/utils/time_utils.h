#pragma once

#include <ctime>
#include <string>

namespace dorado::utils {

std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms);

// Expects the time to be encoded like "2017-09-12T09:50:12.456+00:00" or "2017-09-12T09:50:12Z".
// Time stamp can be specified up to microseconds
time_t get_unix_time_from_string_timestamp(const std::string& time_stamp);

std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms);

std::string adjust_time(const std::string& time_stamp, uint32_t offset);

double time_difference_seconds(const std::string& timestamp1, const std::string& timestamp2);

}  // namespace dorado::utils
