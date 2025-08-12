#pragma once

#include "dorado_version.h"
#include "utils/dev_utils.h"

#ifdef _WIN32
// Unreachable code warnings are emitted from argparse, even though they should be disabled by the
// MSVC /external:W0 setting.  This is a limitation of /external: for some C47XX backend warnings.  See:
// https://learn.microsoft.com/en-us/cpp/build/reference/external-external-headers-diagnostics?view=msvc-170#limitations
#pragma warning(push)
#pragma warning(disable : 4702)
#endif  // _WIN32
#include <argparse/argparse.hpp>
#ifdef _WIN32
#pragma warning(pop)
#endif  // _WIN32

#include <string>

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

inline void parse(argparse::ArgumentParser& parser, const std::vector<std::string>& arguments) {
    parser.add_argument("--devopts")
            .hidden()
            .help("Internal options for testing & debugging, 'key=value' pairs separated by ';'")
            .default_value(std::string(""));
    parser.parse_args(arguments);
    utils::details::extract_dev_options(parser.get<std::string>("--devopts"));
}

inline void parse(argparse::ArgumentParser& parser, int argc, const char* const argv[]) {
    return parse(parser, {argv, argv + argc});
}

}  // namespace dorado::utils::arg_parse