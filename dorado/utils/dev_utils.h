#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace dorado::utils {

namespace details {
extern const char* const g_env_var_name;
extern std::unordered_map<std::string, std::pair<double, bool>> g_dev_options;
void extract_dev_options(const std::string& env_string);
}  // namespace details

// This is intended for dev/testing/debugging purposes, providing a way to influence runtime
// behaviour of Dorado by passing (numerical) values via an environment variable. This is for
// temporary options which are not meant to be exposed as command line options to the user.
//
// Key and value are separated by '=', key/value pairs are separated by ';', e.g.
//
//  $ DORADO_DEV_OPTS="some_opt=42;other_opt=0.5" dorado basecaller ...
//
template <typename T>
T get_dev_opt(const std::string& name, T default_value) {
    if (details::g_dev_options.empty()) {
        const char* env_var = getenv(details::g_env_var_name);
        if (env_var != nullptr) {
            details::extract_dev_options(env_var);
        }
    }

    auto it = details::g_dev_options.find(name);
    if (it != details::g_dev_options.end()) {
        if (it->second.second) {
            // Output debug message the first time we look up a dev variable
            spdlog::debug("{} (environment variable): using '{}' = {}", details::g_env_var_name,
                          name, T(it->second.first));
            it->second.second = false;
        }
        return T(it->second.first);
    }
    return default_value;
}

}  // namespace dorado::utils
