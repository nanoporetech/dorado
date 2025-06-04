#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <unordered_map>

namespace dorado::utils {

namespace details {
extern std::unordered_map<std::string, std::pair<double, bool>> g_dev_options;
void extract_dev_options(const std::string& env_string);
}  // namespace details

// This is intended for dev/testing/debugging purposes, providing a way to influence runtime
// behaviour of Dorado by passing (numerical) values via '--devopts' command line option. This is
// for temporary options which are not meant to be exposed to the end user.
//
// Key and value are separated by '=', key/value pairs are separated by ';', e.g.
//
//  $ dorado basecaller --devopts "some_opt=42;other_opt=0.5" ...
//
template <typename T>
T get_dev_opt(const std::string& name, T default_value) {
    if (auto it = details::g_dev_options.find(name); it != details::g_dev_options.end()) {
        if (it->second.second) {
            // Output debug message the first time we look up a dev variable
            spdlog::debug("DEVOPTS: using '{}' = {}", name, T(it->second.first));
            it->second.second = false;
        }
        return T(it->second.first);
    }
    return default_value;
}

}  // namespace dorado::utils
