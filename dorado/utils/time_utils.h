#pragma once

#include <string>

namespace dorado::utils {

std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms);

time_t get_unix_time_from_string_timestamp(const std::string& time_stamp);

std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms);

std::string adjust_time(const std::string& time_stamp, uint32_t offset);
}  // namespace dorado::utils
