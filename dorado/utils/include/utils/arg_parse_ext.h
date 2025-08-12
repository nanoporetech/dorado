#pragma once

#include "dorado_version.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dorado::utils::arg_parse {

inline bool parse_yes_or_no(const std::string& str) {
    if (str == "yes" || str == "y") {
        return true;
    }
    if (str == "no" || str == "n") {
        return false;
    }
    auto msg = "Unsupported value '" + str + "'; option only accepts '(y)es' or '(n)o'.";
    throw std::runtime_error(msg);
}

template <class T = int64_t>
std::vector<T> parse_string_to_sizes(const std::string& str) {
    std::vector<T> sizes;
    const char* c_str = str.c_str();
    char* p;
    while (true) {
        double x = strtod(c_str, &p);
        if (p == c_str) {
            throw std::runtime_error("Cannot parse size '" + str + "'.");
        }
        if (*p == 'G' || *p == 'g') {
            x *= 1e9;
            ++p;
        } else if (*p == 'M' || *p == 'm') {
            x *= 1e6;
            ++p;
        } else if (*p == 'K' || *p == 'k') {
            x *= 1e3;
            ++p;
        }
        sizes.emplace_back(static_cast<T>(std::round(x)));
        if (*p == ',') {
            c_str = ++p;
            continue;
        } else if (*p == 0) {
            break;
        }
        throw std::runtime_error("Unknown suffix '" + std::string(p) + "'.");
    }
    return sizes;
}

template <class T = uint64_t>
T parse_string_to_size(const std::string& str) {
    return parse_string_to_sizes<T>(str)[0];
}

}  // namespace dorado::utils::arg_parse