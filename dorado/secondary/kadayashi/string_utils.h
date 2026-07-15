#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace kadayashi {

bool stoi_catch(int &dest, const char *s);
bool stod_catch(double &dest, const char *s);

/**
 * @brief Construct a htslib-style region string (1-index, inclusive–inclusive)
 *        from 0-index, inclusive-exclusive coordinates.
 *
 * @param ref_name Reference sequence name.
 * @param start    0-based inclusive start position.
 * @param end      0-based exclusive end position.
 * @return Region string in the format "ref_name:start-end".
 */
std::string create_region_string(const std::string_view ref_name,
                                 const uint32_t start,
                                 const uint32_t end);

bool split_region_string(const std::string &itvl_str,  // 1-index []
                         std::string &tname,
                         uint32_t &start,
                         uint32_t &end  // 0-index [)]
);

}  // namespace kadayashi
