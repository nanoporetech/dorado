#include "time_utils.h"

namespace dorado::utils {

std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms) {
    static std::mutex timestamp_mtx;
    std::unique_lock lock(timestamp_mtx);
    //Convert a time_t (seconds from UNIX epoch) to a timestamp in ISO 8601 (YYYY-MM-DDTHH:MM:SS.sss+00:00) format
    auto time_stamp_s = time_stamp_ms / 1000;
    int num_ms = time_stamp_ms % 1000;
    char buf[32];
    struct tm ts;
    ts = *gmtime(&time_stamp_s);

    std::stringstream ss;
    ss << std::put_time(&ts, "%Y-%m-%dT%H:%M:%S.");
    ss << std::setfill('0') << std::setw(3) << num_ms;  // add ms
    ss << "+00:00";                                     //add zero timezone
    return ss.str();
}

// Expects the time to be encoded like "2017-09-12T9:50:12.456+00:00".
time_t get_unix_time_from_string_timestamp(const std::string& time_stamp) {
    std::stringstream ss(time_stamp);
    std::tm base_time = {};
    ss >> std::get_time(&base_time, "%Y-%m-%dT%H:%M:%S.");

    auto num_ms = std::stoi(time_stamp.substr(20, time_stamp.size() - 26));
    return mktime(&base_time) * 1000 + num_ms;
}

std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms) {
    return utils::get_string_timestamp_from_unix_time(
            get_unix_time_from_string_timestamp(time_stamp) + offset_ms);
}

std::string adjust_time(const std::string& time_stamp, uint32_t offset) {
    // Expects the time to be encoded like "2017-09-12T9:50:12Z".
    // Adds the offset (in seconds) to the timeStamp.
    std::tm base_time = {};
    strptime(time_stamp.c_str(), "%Y-%m-%dT%H:%M:%SZ", &base_time);
    time_t timeObj = mktime(&base_time);
    timeObj += offset;
    std::tm* new_time = gmtime(&timeObj);
    char buff[32];
    strftime(buff, 32, "%FT%TZ", new_time);
    return std::string(buff);
}

}  // namespace dorado::utils
